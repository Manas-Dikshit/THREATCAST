"""Parsers: CSV (generic / CIC-IDS2018 / NetFlow-style) and PCAP via Scapy."""

from .csv_flows import parse_flow_csv, read_raw_csv, records_from_table
from .columns import map_columns, normalize_header
from .pcap_parser import parse_pcap

__all__ = ["parse_flow_csv", "read_raw_csv", "records_from_table", "map_columns", "normalize_header", "parse_pcap"]
