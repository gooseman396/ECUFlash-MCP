import csv
from pathlib import Path

import evo_tuner_mcp as etm


def _write_xml(path: Path, xmlid: str, include: str | None = None, tables: list[str] | None = None, scalings: list[str] | None = None, internalidhex: str | None = None, internalidaddress: str | None = None) -> None:
    parts = ["<rom>"]
    parts.append("  <romid>")
    parts.append(f"    <xmlid>{xmlid}</xmlid>")
    if internalidaddress is not None:
        parts.append(f"    <internalidaddress>{internalidaddress:x}</internalidaddress>")
    if internalidhex is not None:
        parts.append(f"    <internalidhex>{internalidhex}</internalidhex>")
    parts.append("  </romid>")
    if include:
        parts.append(f"  <include>{include}</include>")
    if scalings:
        parts.extend(scalings)
    if tables:
        parts.extend(tables)
    parts.append("</rom>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _make_rom(path: Path, internal_id_hex: str, internal_id_address: int, scalars: dict[int, int]) -> None:
    size = max([internal_id_address + (len(internal_id_hex) // 2)] + [addr + 2 for addr in scalars])
    data = bytearray(size + 16)
    data[internal_id_address : internal_id_address + len(internal_id_hex) // 2] = bytes.fromhex(internal_id_hex)
    for address, value in scalars.items():
        data[address : address + 2] = value.to_bytes(2, byteorder="big", signed=False)
    path.write_bytes(bytes(data))


def _write_log(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_values(path: Path, entries: dict[int, list[int]]) -> None:
    data = bytearray(path.read_bytes())
    for address, values in entries.items():
        offset = address
        for value in values:
            data[offset : offset + 2] = value.to_bytes(2, byteorder="big", signed=False)
            offset += 2
    path.write_bytes(bytes(data))


def test_xml_chain_read_and_write(tmp_path, monkeypatch):
    base_xml = tmp_path / "base.xml"
    top_xml = tmp_path / "top.xml"
    rom = tmp_path / "test.bin"

    scalings = [
        "  <scaling name=\"RPMStatLimit\" units=\"RPM\" toexpr=\"x\" frexpr=\"x\" storagetype=\"uint16\" endian=\"big\"/>",
    ]
    tables = [
        "  <table name=\"Stationary Rev Limiter\" address=\"0x20\" type=\"1D\" scaling=\"RPMStatLimit\"/>",
    ]
    _write_xml(base_xml, "base", tables=tables, scalings=scalings)
    _write_xml(top_xml, "top", include="base", internalidhex="55570306", internalidaddress=0x5002A)

    _make_rom(rom, "55570306", 0x5002A, {0x20: 6500})

    monkeypatch.setattr(etm, "XML_PATHS", [str(tmp_path)])
    monkeypatch.setattr(etm, "ROM_PATH", str(tmp_path))
    monkeypatch.setattr(etm, "OUTPUT_ROM_PATH", str(tmp_path / "out"))
    monkeypatch.setattr(etm, "REQUIRE_BACKUP", False)

    chain = etm.get_definition_chain(rom.name)
    assert any(item.get("xmlid") == "top" for item in chain)
    assert any(item.get("xmlid") == "base" for item in chain)

    tables = etm.list_tables_for_rom(rom.name, "Stationary", limit=10, pretty=True)
    assert any("Stationary Rev Limiter" in row for row in tables)

    table = etm.read_table_for_rom(rom.name, "Stationary Rev Limiter")
    assert table["data"] == [[6500.0]]

    scalar = etm.read_scalar_for_rom(rom.name, "Stationary Rev Limiter")
    assert scalar["value"] == 6500.0

    updated = etm.write_table_for_rom(
        rom.name,
        "Stationary Rev Limiter",
        [[6400.0]],
        output_filename="test_modified.bin",
        overwrite_output=True,
        max_delta=200.0,
    )
    assert "output_path" in updated
    assert updated["metadata_path"] is not None
    assert Path(updated["metadata_path"]).exists()

    updated_scalar = etm.write_scalar_for_rom(
        rom.name,
        "Stationary Rev Limiter",
        6300.0,
        output_filename="test_modified_2.bin",
        overwrite_output=True,
        max_delta=200.0,
    )
    assert "output_path" in updated_scalar
    assert Path(updated_scalar["metadata_path"]).exists()

    profiles = etm.get_allowlist_profiles()
    assert "launch" in profiles["profiles"]
    set_profile = etm.set_allowlist_profile("launch")
    assert set_profile["active_profile"] == "launch"
    cleared = etm.set_allowlist_profile(None)
    assert cleared["active_profile"] is None


def test_launch_log_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(etm, "LOG_PATH", str(tmp_path))

    log_a = tmp_path / "log_a.csv"
    log_b = tmp_path / "log_b.csv"

    rows_a = [
        {
            "LogEntrySeconds": "1.0",
            "RPM": "3000",
            "TPS": "80",
            "Speed": "2",
            "Boost": "10",
            "AFR": "12.0",
            "TimingAdv": "5",
            "KnockSum": "0",
            "IPW": "6",
            "IDC": "20",
            "Load": "150",
        },
        {
            "LogEntrySeconds": "1.1",
            "RPM": "3200",
            "TPS": "82",
            "Speed": "2",
            "Boost": "12",
            "AFR": "11.8",
            "TimingAdv": "4",
            "KnockSum": "0",
            "IPW": "6.5",
            "IDC": "21",
            "Load": "155",
        },
    ]
    rows_b = [
        {
            "LogEntrySeconds": "1.0",
            "RPM": "3000",
            "TPS": "80",
            "Speed": "2",
            "Boost": "14",
            "AFR": "12.2",
            "TimingAdv": "6",
            "KnockSum": "0",
            "IPW": "6",
            "IDC": "20",
            "Load": "150",
        }
    ]

    _write_log(log_a, rows_a)
    _write_log(log_b, rows_b)

    summary = etm.analyze_launch_log(log_a.name)
    assert summary["summary"]["sample_count"] == 2

    comparison = etm.compare_launch_logs(log_a.name, log_b.name)
    assert comparison["diff"]["boost_max"] == 2.0

    extracted = etm.extract_launch_window(log_a.name)
    output_path = tmp_path / extracted["output_filename"]
    assert output_path.exists()
    assert extracted["rows_written"] == 2


def test_rom_compare_axis_and_mapping(tmp_path, monkeypatch):
    base_xml = tmp_path / "base2.xml"
    top_xml = tmp_path / "top2.xml"
    rom_a = tmp_path / "test_a.bin"
    rom_b = tmp_path / "test_b.bin"

    scalings = [
        "  <scaling name=\"AFR\" units=\"AFR\" toexpr=\"x\" frexpr=\"x\" storagetype=\"uint16\" endian=\"big\"/>",
        "  <scaling name=\"Load\" units=\"load\" toexpr=\"x\" frexpr=\"x\" storagetype=\"uint16\" endian=\"big\"/>",
        "  <scaling name=\"RPM\" units=\"rpm\" toexpr=\"x\" frexpr=\"x\" storagetype=\"uint16\" endian=\"big\"/>",
    ]
    tables = [
        "  <table name=\"Test Fuel Map\" address=\"0x100\" type=\"3D\" scaling=\"AFR\">",
        "    <table name=\"Load\" address=\"0x200\" type=\"X Axis\" elements=\"3\" scaling=\"Load\"/>",
        "    <table name=\"RPM\" address=\"0x210\" type=\"Y Axis\" elements=\"2\" scaling=\"RPM\"/>",
        "  </table>",
    ]
    _write_xml(base_xml, "base2", tables=tables, scalings=scalings)
    _write_xml(top_xml, "top2", include="base2", internalidhex="55570306", internalidaddress=0x5002A)

    _make_rom(rom_a, "55570306", 0x5002A, {})
    rom_b.write_bytes(rom_a.read_bytes())

    _write_values(
        rom_a,
        {
            0x200: [1, 2, 3],
            0x210: [1000, 2000],
            0x100: [10, 11, 12, 20, 21, 22],
        },
    )
    _write_values(
        rom_b,
        {
            0x200: [1, 2, 3],
            0x210: [1000, 2000],
            0x100: [10, 11, 12, 20, 21, 23],
        },
    )

    monkeypatch.setattr(etm, "XML_PATHS", [str(tmp_path)])
    monkeypatch.setattr(etm, "ROM_PATH", str(tmp_path))
    monkeypatch.setattr(etm, "LOG_PATH", str(tmp_path))

    axis_bundle = etm.read_table_with_axes_for_rom(rom_a.name, "Test Fuel Map")
    assert axis_bundle["x_axis"]["values"] == [1.0, 2.0, 3.0]
    assert axis_bundle["y_axis"]["values"] == [1000.0, 2000.0]

    search = etm.search_tables_for_rom(rom_a.name, "fuel", limit=5, pretty=True)
    assert any("Test Fuel Map" in row for row in search)

    diff = etm.compare_roms(rom_a.name, rom_b.name, contains="fuel")
    assert diff["changed_count"] == 1

    log_rows = [
        {
            "LogEntrySeconds": "1.0",
            "RPM": "1000",
            "TPS": "80",
            "Speed": "2",
            "Load": "2",
        },
        {
            "LogEntrySeconds": "1.1",
            "RPM": "2000",
            "TPS": "80",
            "Speed": "2",
            "Load": "3",
        },
    ]
    _write_log(tmp_path / "map_log.csv", log_rows)

    mapping = etm.map_log_to_table(
        rom_a.name,
        "Test Fuel Map",
        "map_log.csv",
        rpm_min=500,
        rpm_max=2500,
    )
    assert mapping["total_cells"] >= 1
