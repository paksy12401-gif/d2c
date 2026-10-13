from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class DwgConversionError(RuntimeError):
    pass


def _find_oda_converter() -> Path | None:
    configured = os.getenv("ODA_FILE_CONVERTER")
    candidates = [
        configured,
        shutil.which("ODAFileConverter"),
        shutil.which("ODAFileConverter.exe"),
        r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files\ODA\ODAFile Converter\ODAFileConverter.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def _find_accoreconsole() -> Path | None:
    configured = os.getenv("ACCORECONSOLE")
    candidates = [configured, shutil.which("accoreconsole"), shutil.which("accoreconsole.exe")]
    autodesk_root = Path(r"C:\Program Files\Autodesk")
    if autodesk_root.is_dir():
        candidates.extend(sorted(autodesk_root.glob("AutoCAD */accoreconsole.exe"), reverse=True))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def _autocad_to_dxf(source: Path, target: Path, executable: Path | None = None) -> Path:
    executable = executable or _find_accoreconsole()
    if not executable:
        raise DwgConversionError("AutoCAD Core Console을 찾지 못했습니다.")
    target.parent.mkdir(parents=True, exist_ok=True)
    command_target = target.resolve().as_posix()
    script = target.parent / f"{target.stem}_convert.scr"
    script.write_text(
        "FILEDIA\n"
        "0\n"
        "_.SAVEAS\n"
        "_DXF\n"
        "16\n"
        f"{command_target}\n",
        encoding="ascii",
    )
    command = [
        str(executable),
        "/i",
        str(source),
        "/s",
        str(script),
        "/l",
        "en-US",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=180)
    if completed.returncode == 0 and target.exists():
        return target
    stderr = completed.stderr.decode("utf-8", errors="replace")[-800:]
    stdout = completed.stdout.decode("utf-8", errors="replace")[-800:]
    raise DwgConversionError(
        "AutoCAD Core Console이 DXF 변환에 실패했습니다. "
        f"종료 코드={completed.returncode}, 출력={stdout}, 오류={stderr}"
    )


def ensure_dxf(source: Path, workspace: Path) -> Path:
    source = source.resolve()
    if source.suffix.lower() == ".dxf":
        return source
    if source.suffix.lower() != ".dwg":
        raise DwgConversionError(f"지원하지 않는 형식입니다: {source.suffix}")

    workspace.mkdir(parents=True, exist_ok=True)
    autocad = _find_accoreconsole()
    if autocad:
        return _autocad_to_dxf(source, workspace / f"{source.stem}.dxf", autocad)

    oda = _find_oda_converter()
    if oda:
        output_dir = workspace / f"{source.stem}_oda"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(oda),
            str(source.parent),
            str(output_dir),
            "ACAD2018",
            "DXF",
            "0",
            "1",
            source.name,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        matches = list(output_dir.glob(f"{source.stem}.dxf"))
        if completed.returncode == 0 and matches:
            return matches[0]
        raise DwgConversionError(
            "ODA File Converter가 DWG를 변환하지 못했습니다. "
            f"종료 코드={completed.returncode}, 오류={completed.stderr[-500:]}"
        )

    libre = shutil.which("dwg2dxf") or shutil.which("dwg2dxf.exe")
    if libre:
        target = workspace / f"{source.stem}.dxf"
        attempts = [
            [libre, str(source), "-o", str(target)],
            [libre, "-o", str(target), str(source)],
        ]
        for command in attempts:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if completed.returncode == 0 and target.exists():
                return target

    raise DwgConversionError(
        "DWG를 읽으려면 AutoCAD Core Console, ODA File Converter 또는 LibreDWG의 dwg2dxf가 필요합니다. "
        "ACCORECONSOLE/ODA_FILE_CONVERTER 환경변수에 실행 파일 경로를 지정하거나 DXF로 저장해 주세요."
    )
