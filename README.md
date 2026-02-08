# ECUFlash-MCP
A focused MCP server for Evo tuning. It reads and writes ROM tables via EcuFlash XML metadata, supports TephraMOD V3 features, and adds launch-log analysis, ROM diffing, axis-aware table reads, and safe write guardrails with profiles and metadata snapshots.

## MCP Server Setup

### Requirements
- Python 3.12+.
- EcuFlash ROM metadata XMLs installed (default: `C:\Program Files (x86)\OpenECU\EcuFlash\rommetadata\mitsubishi\evo`).
- ROM files stored in the ROM path configured below.

### Configuration (Environment Variables)
Set these before launching the MCP server.

- `EVO_ROM_PATH` (default: `./Roms`)
- `EVO_OUTPUT_ROM_PATH` (default: `./Roms/modified`)
- `EVO_LOG_PATH` (default: `./logs`)
- `EVO_XML_PATHS` (default: `C:\Program Files (x86)\OpenECU\EcuFlash\rommetadata\mitsubishi\evo`)
- `EVO_TABLE_ALLOWLIST` (optional, comma-separated table name patterns)
- `EVO_ALLOWLIST_PROFILE` (optional, one of: `launch`, `boost`, `fuel`, `timing`, `mivec`, `rev`)
- `EVO_DEFAULT_USE_MAX_DELTA` (optional, `true` to enable default max-delta guardrails)
- `EVO_REQUIRE_BACKUP` (default: `true`)
- `EVO_CONFIG_PATH` (optional, path to `mcp_config.json` for defaults)
- `EVO_CONFIG_STRICT` (optional, `true` to fail fast on invalid config)

Optional config sections in `mcp_config.json`:
- `guardrail_profiles`
- `log_signal_sets`
- `launch_health`
- `torque_risk`
- `tuning_usecases`

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
- Usecase discovery: `list_tuning_usecases` -> `get_tuning_usecase` -> `list_usecase_tables_for_rom`.
- Table inspect: `find_table` -> `read_table_with_axes_for_rom`.
- Launch analysis: `analyze_launch_log` -> `extract_launch_window` -> `map_log_to_table`.
- Launch health report: `analyze_launch_health` -> `compare_launch_logs`.
- Boost control sanity: `check_boost_control_consistency`.
- Low-RPM torque risk: `estimate_low_rpm_torque_risk`.
- Safe write preview: `preview_write_table_for_rom` or `preview_write_cell_for_rom` before `write_table_for_rom`.
- Cache refresh after XML edits: `refresh_xml_cache`.
- Direct boost blobs (RAX3): `read_blob_for_rom` with explicit length/data_type.
- Blob bitfields: `read_blob_bits_for_rom` for ECU options and mode flags.
- Blob decoding: `decode_blob_table_for_rom` for known bloblist scalings.
- Definition audit: `diagnose_rom_definitions` for missing rows/axes/scalings.

### AI-Friendly Helpers
- `get_current_context` provides paths, allowlist profile, and active settings.
- `list_common_tables_for_rom` reduces table hunting by profile.
- `list_tuning_usecases` and `list_usecase_tables_for_rom` cover boost/fuel/timing workflows.
- `check_log_signals` reports missing signals for launch/boost/fuel/timing analyses.
- `analyze_launch_health` combines log and ROM checks into a single report.
- `estimate_low_rpm_torque_risk` flags risky low-RPM boost conditions.
- `check_boost_control_consistency` highlights mismatched or missing boost control tables.
- `compare_table_between_roms` summarizes deltas without writing changes.
- `find_table` returns axis metadata for exact matches.
- `preview_write_table_for_rom` reports deltas without writing.
- Axis inference uses a registry for common RPM/Load/Throttle axis addresses.

### Max-Delta Guardrails
- Default max-delta blocking is disabled. Set `max_delta` or `use_default_max_delta=true` on write/preview calls when you want guardrails.

## Risks and Disclaimer
- Use at your own risk. Calibration changes can damage engines, drivetrains, or emissions systems if used incorrectly.
- Logs, tables, and recommendations are informational only and may be incomplete or inaccurate.
- You are solely responsible for any changes you apply to your vehicle and for validating results safely.
- By using this MCP server, you agree that the author is not responsible or liable for any issues, damage, or losses related to its use.
