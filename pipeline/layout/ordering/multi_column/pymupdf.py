"""Utilities for detecting multi-column page regions with PyMuPDF."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore

__all__ = ["column_boxes"]

_MIN_NBLOCK_LENGTH = 2
_BOTTOM_TOLERANCE = 10

PyMuPDFModule = Any
PyMuPDFPage = Any
PyMuPDFRect = Any
PyMuPDFIRect = Any


def column_boxes(
    page: PyMuPDFPage,
    footer_margin: int = 50,
    header_margin: int = 50,
    no_image_text: bool = True,
) -> list[PyMuPDFIRect]:
    """Determine bounding boxes that wrap text columns on a page."""

    fitz_module = _ensure_pymupdf_available()

    clip = _create_clip_rect(fitz_module, page, footer_margin, header_margin)
    path_bboxes = _extract_path_bboxes(fitz_module, page)
    img_bboxes = _extract_image_bboxes(fitz_module, page)
    text_bboxes, vert_bboxes = _extract_text_bboxes(
        fitz_module,
        page,
        clip,
        img_bboxes,
        no_image_text,
    )

    if not text_bboxes:
        return []

    text_bboxes = _sort_text_bboxes(text_bboxes, path_bboxes)
    text_bboxes = _extend_right(
        text_bboxes,
        int(page.rect.width),
        path_bboxes,
        vert_bboxes,
        img_bboxes,
    )

    if not text_bboxes:
        return []

    joined_bboxes = _join_bboxes(text_bboxes, path_bboxes, vert_bboxes)
    return _clean_nblocks(joined_bboxes)


def _ensure_pymupdf_available() -> PyMuPDFModule:
    if fitz is None:  # pragma: no cover - guard optional dependency
        raise ImportError("PyMuPDF (fitz) is required for column detection")
    return cast(PyMuPDFModule, fitz)


def _create_clip_rect(
    _fitz: PyMuPDFModule,
    page: PyMuPDFPage,
    footer_margin: int,
    header_margin: int,
) -> PyMuPDFRect:
    clip = +page.rect
    clip.y1 -= footer_margin
    clip.y0 += header_margin
    return clip


def _extract_path_bboxes(_fitz: PyMuPDFModule, page: PyMuPDFPage) -> list[PyMuPDFIRect]:
    path_bboxes: list[PyMuPDFIRect] = []
    for path in page.get_drawings():
        path_bboxes.append(cast(PyMuPDFIRect, path["rect"].irect))
    path_bboxes.sort(key=lambda rect: (rect.y0, rect.x0))
    return path_bboxes


def _extract_image_bboxes(_fitz: PyMuPDFModule, page: PyMuPDFPage) -> list[PyMuPDFIRect]:
    img_bboxes: list[PyMuPDFIRect] = []
    for item in page.get_images():
        img_bboxes.extend(cast(Sequence[PyMuPDFIRect], page.get_image_rects(item[0])))
    return img_bboxes


def _extract_text_bboxes(
    fitz_module: PyMuPDFModule,
    page: PyMuPDFPage,
    clip: PyMuPDFRect,
    img_bboxes: Sequence[PyMuPDFIRect],
    no_image_text: bool,
) -> tuple[list[PyMuPDFIRect], list[PyMuPDFIRect]]:
    bboxes: list[PyMuPDFIRect] = []
    vert_bboxes: list[PyMuPDFIRect] = []

    blocks = page.get_text(
        "dict",
        flags=fitz_module.TEXTFLAGS_TEXT,
        clip=clip,
    )["blocks"]

    for block in blocks:
        bbox = fitz_module.IRect(block["bbox"])

        if no_image_text and _in_bbox(bbox, img_bboxes):
            continue

        first_line = block["lines"][0]
        if first_line["dir"] != (1, 0):
            vert_bboxes.append(bbox)
            continue

        srect = fitz_module.EMPTY_IRECT()
        for line in block["lines"]:
            line_bbox = fitz_module.IRect(line["bbox"])
            text = "".join(span["text"].strip() for span in line["spans"])
            if len(text) > 1:
                srect |= line_bbox

        bbox = +srect
        if not bbox.is_empty:
            bboxes.append(bbox)

    return bboxes, vert_bboxes


def _sort_text_bboxes(
    bboxes: list[PyMuPDFIRect],
    path_bboxes: Sequence[PyMuPDFIRect],
) -> list[PyMuPDFIRect]:
    bboxes.sort(key=lambda rect: (_in_bbox(rect, path_bboxes), rect.y0, rect.x0))
    return bboxes


def _extend_right(
    bboxes: list[PyMuPDFIRect],
    width: int,
    path_bboxes: Sequence[PyMuPDFIRect],
    vert_bboxes: Sequence[PyMuPDFIRect],
    img_bboxes: Sequence[PyMuPDFIRect],
) -> list[PyMuPDFIRect]:
    for index, bbox in enumerate(bboxes):
        if _in_bbox(bbox, path_bboxes):
            continue

        if _in_bbox(bbox, img_bboxes):
            continue

        temp_rect = +bbox
        temp_rect.x1 = width

        combined_blocks = list(path_bboxes) + list(vert_bboxes) + list(img_bboxes)

        if _intersects_bboxes(temp_rect, combined_blocks):
            continue

        if _can_extend(temp_rect, bbox, bboxes, vert_bboxes):
            bboxes[index] = temp_rect

    return [rect for rect in bboxes if rect is not None]


def _join_bboxes(
    bboxes: list[PyMuPDFIRect],
    path_bboxes: Sequence[PyMuPDFIRect],
    vert_bboxes: Sequence[PyMuPDFIRect],
) -> list[PyMuPDFIRect]:
    nblocks: list[PyMuPDFIRect] = [bboxes[0]]
    remaining: list[PyMuPDFIRect | None] = list(bboxes[1:])

    for index, bbox in enumerate(remaining):
        if bbox is None:
            continue

        merged, merge_index, temp_rect = _find_merge_candidate(
            bbox,
            nblocks,
            path_bboxes,
            vert_bboxes,
        )

        if not merged:
            nblocks.append(bbox)
            merge_index = len(nblocks) - 1
            temp_rect = nblocks[merge_index]

        if not _can_extend(temp_rect, bbox, remaining, vert_bboxes):
            nblocks.append(bbox)
        else:
            nblocks[merge_index] = temp_rect

        remaining[index] = None

    return nblocks


def _find_merge_candidate(
    bbox: PyMuPDFIRect,
    nblocks: Sequence[PyMuPDFIRect],
    path_bboxes: Sequence[PyMuPDFIRect],
    vert_bboxes: Sequence[PyMuPDFIRect],
) -> tuple[bool, int, PyMuPDFIRect]:
    for new_index, new_bbox in enumerate(nblocks):
        if new_bbox.x1 < bbox.x0 or bbox.x1 < new_bbox.x0:
            continue

        if _in_bbox(new_bbox, path_bboxes) != _in_bbox(bbox, path_bboxes):
            continue

        candidate = bbox | new_bbox
        if _can_extend(candidate, new_bbox, nblocks, vert_bboxes):
            return True, new_index, candidate

    return False, 0, bbox


def _clean_nblocks(nblocks: list[PyMuPDFIRect]) -> list[PyMuPDFIRect]:
    blen = len(nblocks)
    if blen < _MIN_NBLOCK_LENGTH:
        return nblocks

    start = blen - 1
    for idx in range(start, -1, -1):
        bb1 = nblocks[idx]
        bb0 = nblocks[idx - 1]
        if bb0 == bb1:
            del nblocks[idx]

    y1 = nblocks[0].y1
    i0 = 0
    i1 = -1

    for idx in range(1, len(nblocks)):
        bbox1 = nblocks[idx]
        if abs(bbox1.y1 - y1) > _BOTTOM_TOLERANCE:
            if i1 > i0:
                nblocks[i0 : i1 + 1] = sorted(
                    nblocks[i0 : i1 + 1],
                    key=lambda rect: rect.x0,
                )
            y1 = bbox1.y1
            i0 = idx
        i1 = idx

    if i1 > i0:
        nblocks[i0 : i1 + 1] = sorted(nblocks[i0 : i1 + 1], key=lambda rect: rect.x0)
    return nblocks


def _in_bbox(bb: PyMuPDFIRect, target_boxes: Sequence[PyMuPDFIRect]) -> int:
    for index, bbox in enumerate(target_boxes):
        if bb in bbox:
            return index + 1
    return 0


def _intersects_bboxes(bb: PyMuPDFIRect, bboxes_to_check: Sequence[PyMuPDFIRect]) -> bool:
    for bbox in bboxes_to_check:
        if not (bb & bbox).is_empty:
            return True
    return False


def _can_extend(
    temp: PyMuPDFIRect,
    bbox: PyMuPDFIRect,
    bbox_list: Sequence[PyMuPDFIRect | None],
    vert_bboxes: Sequence[PyMuPDFIRect],
) -> bool:
    for existing in bbox_list:
        if not _intersects_bboxes(temp, vert_bboxes) and (
            existing is None or existing is bbox or (temp & existing).is_empty
        ):
            continue
        return False

    return True
