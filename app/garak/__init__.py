"""Garak security-scanner integration.

garak (github.com/NVIDIA/garak) is an LLM vulnerability scanner. This package
runs it as a subprocess against the running app via garak's REST generator, so
any provider/model the app exposes through get_llm() can be red-teamed -- either
the raw LLM or the full RAG pipeline. See runner.run_scan.
"""
