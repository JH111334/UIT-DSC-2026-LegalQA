# QA offline

Nhánh này sở hữu chuẩn hóa tối thiểu, grouping chống leakage và split QA. Hàm build
được export từ `QA/__init__.py`; `Offline/release_builder.py` ghép QA với corpus và
diagnostics thành một release bất biến.
