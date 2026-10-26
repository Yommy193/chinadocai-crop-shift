from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import mimetypes
import re
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn

from .data import _square_crop
from .infer import (
    _load_sequence,
    _load_vision,
    _standalone_square,
    order_tokens,
    recognize,
    rerank_recognition_candidates,
    suggest_missing,
)
from .models import OracleSequenceModel
from .utils import project_root, resolve_project_path


MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
MAX_IMAGES = 32
IDENTIFIER = re.compile(r"^[a-z0-9]{10}$")
REFERENCE_FILE = re.compile(r"^[a-z0-9]{10}\.png$")


class ClientError(ValueError):
    pass


@dataclass
class AppState:
    device: torch.device
    vision: nn.Module
    sequence: OracleSequenceModel
    vocabulary: dict[str, Any]
    reference_root: Path
    web_root: Path
    inference_lock: threading.Lock

    @property
    def token_to_label(self) -> dict[int, str]:
        return {int(token): label for label, token in self.vocabulary["label_to_token"].items()}

    def reference_for(self, label: str) -> tuple[Path | None, str | None]:
        if not IDENTIFIER.fullmatch(label):
            return None, None
        directory = self.reference_root / label
        if not directory.is_dir():
            return None, None
        match = next(iter(sorted(directory.glob("*.png"))), None)
        if match is None:
            return None, None
        return match, f"/reference/{quote(label)}/{quote(match.name)}"


def load_state(
    vision_checkpoint: str,
    sequence_checkpoint: str,
    reference_dir: str,
    cpu: bool = False,
) -> AppState:
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    vision, vision_vocabulary = _load_vision(resolve_project_path(vision_checkpoint), device)
    sequence, sequence_vocabulary = _load_sequence(resolve_project_path(sequence_checkpoint), device)
    if vision_vocabulary["labels"] != sequence_vocabulary["labels"]:
        raise RuntimeError("视觉模型和序列模型的字头词表不一致")
    return AppState(
        device=device,
        vision=vision,
        sequence=sequence,
        vocabulary=vision_vocabulary,
        reference_root=resolve_project_path(reference_dir).resolve(),
        web_root=(project_root() / "web").resolve(),
        inference_lock=threading.Lock(),
    )


def decode_data_image(data: str) -> Image.Image:
    if not isinstance(data, str) or not data:
        raise ClientError("图片数据为空")
    encoded = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ClientError("图片不是有效的 Base64 数据") from exc
    if not raw:
        raise ClientError("图片数据为空")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ClientError(f"单张图片不能超过 {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            image = ImageOps.exif_transpose(image)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ClientError("图片像素过大，请先缩小后再上传")
            return _standalone_square(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ClientError("无法读取图片，请使用 PNG、JPG、WebP 或 BMP") from exc


def validate_payload(payload: Any) -> tuple[list[Image.Image], list[str], bool, bool, int | None, int, bool, bool]:
    if not isinstance(payload, dict):
        raise ClientError("请求格式错误")
    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ClientError("请至少拖入一张单字图片")
    if len(raw_images) > MAX_IMAGES:
        raise ClientError(f"一次最多处理 {MAX_IMAGES} 张图片")

    images: list[Image.Image] = []
    names: list[str] = []
    for index, item in enumerate(raw_images):
        if not isinstance(item, dict):
            raise ClientError(f"第 {index + 1} 张图片格式错误")
        images.append(decode_data_image(item.get("data", "")))
        name = item.get("name")
        names.append(str(name).strip()[:180] if name else f"图片 {index + 1}")

    auto_sort = bool(payload.get("auto_sort", True))
    complete_missing = bool(payload.get("complete_missing", False))
    fixed_position = payload.get("missing_position")
    if fixed_position in ("", None):
        fixed_position = None
    elif isinstance(fixed_position, bool):
        raise ClientError("缺字位置必须是整数")
    else:
        try:
            fixed_position = int(fixed_position)
        except (TypeError, ValueError) as exc:
            raise ClientError("缺字位置必须是整数") from exc
    if fixed_position is not None and not 0 <= fixed_position <= len(images):
        raise ClientError(f"缺字位置必须在 0 到 {len(images)} 之间")
    if not complete_missing:
        fixed_position = None

    try:
        top_k = int(payload.get("top_k", 5))
    except (TypeError, ValueError) as exc:
        raise ClientError("候选数量必须是整数") from exc
    if not 1 <= top_k <= 10:
        raise ClientError("候选数量必须在 1 到 10 之间")
    # The deployed CNN was selected with the original view; the former ViT
    # four-view ensemble remains an optional compatibility mode.
    robust_tta = bool(payload.get("robust_tta", False))
    joint_candidate_rerank = bool(payload.get("joint_candidate_rerank", False))
    if joint_candidate_rerank and not auto_sort:
        raise ClientError("联合候选重排需要开启模型自动排序")
    return images, names, auto_sort, complete_missing, fixed_position, top_k, robust_tta, joint_candidate_rerank


def predict_payload(state: AppState, payload: Any) -> dict[str, Any]:
    images, names, auto_sort, complete_missing, fixed_position, top_k, robust_tta, joint_candidate_rerank = validate_payload(payload)
    if len(images) > state.sequence.config.max_length:
        raise ClientError(f"当前模型一次最多排序 {state.sequence.config.max_length} 个字")
    if complete_missing and len(images) + 1 > state.sequence.config.max_length:
        raise ClientError(
            f"开启缺字推测时最多输入 {state.sequence.config.max_length - 1} 个可见字"
        )

    with state.inference_lock:
        tokens, recognition = recognize(
            state.vision, state.vocabulary, images, state.device, max(top_k, 3) if joint_candidate_rerank else top_k, robust_tta=robust_tta
        )
        joint_result = None
        if joint_candidate_rerank:
            joint_result = rerank_recognition_candidates(state.sequence, state.vocabulary, recognition, state.device)
            tokens, order, sort_scores = joint_result.tokens, joint_result.order, joint_result.sort_scores
        elif auto_sort:
            order, sort_scores = order_tokens(state.sequence, tokens, state.device)
        else:
            order = list(range(len(tokens)))
            sort_scores = [float(index) for index in order]

        token_to_label = state.token_to_label
        ordered_tokens = [tokens[index] for index in order]
        completion: list[dict[str, Any]] = []
        if complete_missing:
            completion = suggest_missing(
                state.sequence,
                ordered_tokens,
                token_to_label,
                state.device,
                top_k,
                fixed_position,
            )

    for index, row in enumerate(recognition):
        row["input_index"] = index
        row["input"] = names[index]
        for candidate in row["candidates"]:
            _, candidate["reference_url"] = state.reference_for(candidate["label"])
        if joint_result is not None:
            selected_label = token_to_label[tokens[index]]
            row["joint_selected_label"] = selected_label
            for candidate in row["candidates"]:
                candidate["selected_by_joint"] = candidate["label"] == selected_label

    ordered = []
    for rank, index in enumerate(order):
        label = token_to_label[tokens[index]]
        selected_candidate = next(
            (candidate for candidate in recognition[index]["candidates"] if candidate["label"] == label),
            recognition[index]["candidates"][0],
        )
        _, reference_url = state.reference_for(label)
        ordered.append(
            {
                "rank": rank,
                "original_index": index,
                "input": names[index],
                "label": label,
                "probability": selected_candidate["probability"],
                "visual_top1_label": recognition[index]["candidates"][0]["label"],
                "reference_url": reference_url,
                "sort_score": sort_scores[index],
            }
        )

    for suggestion in completion:
        _, suggestion["reference_url"] = state.reference_for(suggestion["label"])

    return {
        "device": str(state.device),
        "input_count": len(images),
        "auto_sorted": auto_sort,
        "robust_tta": robust_tta,
        "joint_candidate_rerank": {
            "applied": joint_result is not None,
            "candidate_top_k": 3,
            "beam_width": 8,
            "sequence_weight": 0.6,
            "beam_rank_before_rerank": joint_result.beam_rank_before_rerank if joint_result else None,
            "candidate_count": joint_result.candidate_count if joint_result else None,
            "notes": "研究功能：排序在两域有证据支持；拓片缺字候选有提升，摹本缺字收益尚无显著性。",
        },
        "recognition": recognition,
        "ordered": ordered,
        "missing_suggestions": completion,
        "notes": [
            "字头标签是数据集内部 ID，参考字形图片用于人工核对",
            "未知缺字位置按掩码置信度检索，不代表该处一定存在缺字",
        ],
    }


def demo_payload() -> dict[str, Any]:
    """Return locally cropped, deliberately shuffled H51 test glyphs for one-click UI exploration."""
    demo_dir = project_root() / "artifacts" / "demo_h51"
    plate_path = project_root() / "data" / "processed" / "images" / "facsimile" / "h00051.jpg"
    boxes_path = demo_dir / "H51_group0_boxes_shuffled.json"
    if not plate_path.is_file() or not boxes_path.is_file():
        raise FileNotFoundError("项目内演示案例不可用")
    with boxes_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    boxes = metadata.get("boxes")
    if not isinstance(boxes, list) or not boxes or len(boxes) > MAX_IMAGES:
        raise ValueError("项目内演示案例格式无效")

    images = []
    with Image.open(plate_path) as plate:
        plate.load()
        plate = ImageOps.exif_transpose(plate).convert("RGB")
        for index, box in enumerate(boxes, start=1):
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("项目内演示案例的字框无效")
            crop = _square_crop(plate, [int(value) for value in box])
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG", optimize=True)
            images.append(
                {
                    "name": f"H51_打乱字_{index:02d}.png",
                    "data": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                }
            )
    return {
        "title": "H51 保留测试集示例（8 个字，已打乱）",
        "note": "这是项目内测试样例，用于体验流程；识别、排序结果仍应结合参考字形人工核对。",
        "images": images,
    }


class OracleWebHandler(BaseHTTPRequestHandler):
    server_version = "OracleViT/0.1"

    @property
    def state(self) -> AppState:
        return self.server.state  # type: ignore[attr-defined]

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, cache: bool = False) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(
                {
                    "ready": True,
                    "device": str(self.state.device),
                    "classes": len(self.state.vocabulary["labels"]),
                    "max_images": min(MAX_IMAGES, self.state.sequence.config.max_length),
                    "max_images_with_completion": min(
                        MAX_IMAGES, self.state.sequence.config.max_length - 1
                    ),
                }
            )
            return
        if path == "/api/demo":
            try:
                self._send_json(demo_payload())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if path.startswith("/reference/"):
            parts = [unquote(part) for part in path.split("/") if part]
            if len(parts) != 3 or parts[0] != "reference":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            _, label, filename = parts
            if not IDENTIFIER.fullmatch(label) or not REFERENCE_FILE.fullmatch(filename):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            target = (self.state.reference_root / label / filename).resolve()
            if not target.is_relative_to(self.state.reference_root):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_file(target, cache=True)
            return

        static_files = {
            "/": "index.html",
            "/index.html": "index.html",
            "/app.js": "app.js",
            "/styles.css": "styles.css",
            "/favicon.svg": "favicon.svg",
        }
        filename = static_files.get(path)
        if filename is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(self.state.web_root / filename)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "请求长度无效"}, HTTPStatus.BAD_REQUEST)
            return
        if not 0 < content_length <= MAX_REQUEST_BYTES:
            self._send_json({"error": "请求为空或过大"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            result = predict_payload(self.state, payload)
            self._send_json(result)
        except ClientError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "请求不是有效的 JSON"}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - final safety boundary for the local UI
            print(f"推理失败: {exc}", flush=True)
            self._send_json({"error": "推理失败，请查看启动窗口中的错误信息"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}", flush=True)


class OracleWebServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: AppState) -> None:
        super().__init__(address, OracleWebHandler)
        self.state = state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动甲骨文识别、排序和缺字推测本地网页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--vision-checkpoint", default="artifacts/checkpoints/vision_best.pt")
    parser.add_argument("--sequence-checkpoint", default="artifacts/checkpoints/sequence_best.pt")
    parser.add_argument("--reference-dir", default="data/processed/glyph_reference_full")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("端口必须在 1 到 65535 之间")
    print("正在加载视觉模型与序列模型……", flush=True)
    state = load_state(
        args.vision_checkpoint,
        args.sequence_checkpoint,
        args.reference_dir,
        cpu=args.cpu,
    )
    server = OracleWebServer((args.host, args.port), state)
    url = f"http://{args.host}:{args.port}/"
    print(f"模型已就绪：{state.device}，{len(state.vocabulary['labels'])} 个字头", flush=True)
    print(f"网页地址：{url}", flush=True)
    print("关闭此窗口或按 Ctrl+C 可停止服务。", flush=True)
    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n正在停止网页服务……", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
