import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from ads.data import read_rows
from ads.parquet_data import COLUMNS, prepare_parquet


@unittest.skipUnless(find_spec("pyarrow"), "requires the Criteo extra")
class ParquetSplitTest(unittest.TestCase):
    def test_no_day_overlap_and_deterministic_sampling(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input"
            source.mkdir()
            for day in range(15, 20):
                columns = {k: [day]*40 for k in COLUMNS[1:14]}
                columns.update({k: ["a", None]*20 for k in COLUMNS[14:]})
                columns["label"] = [0, 1]*20
                pq.write_table(pa.table(columns), source / f"2015-02-{day}.parquet")
            out, repeat = Path(tmp) / "a", Path(tmp) / "b"
            report = prepare_parquet(source, out, 101, 42)
            self.assertEqual(report, prepare_parquet(source, repeat, 101, 42))
            self.assertEqual(sum(x["sample_rows"] for x in report["windows"].values()), 101)
            for split, days in (("train", {15, 16, 17}), ("validation", {18}), ("test", {19})):
                file = out / f"{split}.tsv"
                rows = list(read_rows(file))
                self.assertEqual({int(row[1]) for row in rows}, days)
                self.assertEqual({row[14] for row in rows}, {"a", ""})
                self.assertEqual(file.read_bytes(), (repeat / file.name).read_bytes())
            all_rows = prepare_parquet(source, Path(tmp) / "all", None, 42)
            self.assertEqual(all_rows["windows"]["test"]["sample_rows"], 40)
