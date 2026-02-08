# ECUFlash-MCP
A focused MCP server for Evo tuning. It reads and writes ROM tables via EcuFlash XML metadata, supports TephraMOD V3 features, and adds launch-log analysis, ROM diffing, axis-aware table reads, and safe write guardrails with profiles and metadata snapshots.

## MCP Server Setup

### Requirements
- Python 3.12+.
- EcuFlash ROM metadata XMLs installed (default: `C:\Program Files (x86)\OpenECU\EcuFlash\rommetadata\mitsubishi\evo`).
- ROM files stored in the ROM path configured below.

### Configuration (Environment Variables)
Set these before launching the MCP server.

- `EVO_ROM_PATH` (default: `C:\Users\Joey\Desktop\AIEvoTunerMCP\Roms`)
- `EVO_OUTPUT_ROM_PATH` (default: `C:\Users\Joey\Desktop\AIEvoTunerMCP\Roms\modified`)
- `EVO_LOG_PATH` (default: `C:\Users\Joey\Documents\EvoScan v2.9\SavedDataLogs`)
- `EVO_XML_PATHS` (default: `C:\Program Files (x86)\OpenECU\EcuFlash\rommetadata\mitsubishi\evo`)
- `EVO_TABLE_ALLOWLIST` (optional, comma-separated table name patterns)
- `EVO_ALLOWLIST_PROFILE` (optional, one of: `launch`, `boost`, `fuel`, `timing`, `mivec`, `rev`)
- `EVO_REQUIRE_BACKUP` (default: `true`)

### Run
From the repo root:

```powershell
# Activate venv if you are using one
python evo_tuner_mcp.py
```

If you are launching via an MCP client, point it at this server entry point and pass the same environment variables.

### Quick sanity checks
- `identify_rom` to confirm XML definition matching.
- `list_tables_for_rom` to verify table discovery.
- `read_table_with_axes_for_rom` to confirm axes and scaling behavior.

### Examples (Common Flows)
- ROM discovery: `identify_rom` -> `get_definition_chain` -> `list_common_tables_for_rom`.
- Table inspect: `find_table` -> `read_table_with_axes_for_rom`.
- Launch analysis: `analyze_launch_log` -> `extract_launch_window` -> `map_log_to_table`.
- Safe write preview: `preview_write_table_for_rom` or `preview_write_cell_for_rom` before `write_table_for_rom`.
- Cache refresh after XML edits: `refresh_xml_cache`.

### AI-Friendly Helpers
- `get_current_context` provides paths, allowlist profile, and active settings.
- `list_common_tables_for_rom` reduces table hunting by profile.
- `find_table` returns axis metadata for exact matches.
- `preview_write_table_for_rom` reports deltas without writing.

## Risks and Disclaimer
- Use at your own risk. Calibration changes can damage engines, drivetrains, or emissions systems if used incorrectly.
- Logs, tables, and recommendations are informational only and may be incomplete or inaccurate.
- You are solely responsible for any changes you apply to your vehicle and for validating results safely.
- By using this MCP server, you agree that the author is not responsible or liable for any issues, damage, or losses related to its use.
