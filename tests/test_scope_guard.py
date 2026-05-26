from __future__ import annotations

import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_GUARDED_FILES = (
    "requirements_data_pipeline.txt",
    "scripts/rebuild_etf_lof_data.py",
    "scripts/run_pyfinance_metrics_factory.py",
)
EXPECTED_GUARDED_FILE_SHA256 = {
    'data/metrics_factory/core13_all_metrics_features.parquet': "7b31ca4ae4c0da03f3fb2346beba0743bc019269e256875338284b4dcba8899e",
    'data/processed/core13_etf_lof_daily_nav_long.parquet': "70857ff4956a199a8fd5486e3f6dfabdf52d36c4de4b8ae322f12e7727b3ce38",
    'data/processed/core13_etf_lof_daily_panel.parquet': "83ca942b4b6e15eefa802755642aa957c387eee452550711501d80eea81b00ef",
    'data/processed/core13_etf_lof_fund_nav_tushare_long.parquet': "b3ba70f78d6b8206d556082f38eca3e4cfa038256d3fcbb6e45f6a9b20e09057",
    'data/processed/core13_wide_acc_nav.parquet': "881be2a21deb05f232d8144010f2659b6bcd7642cf923af5daa4b3eba07cc639",
    'data/processed/core13_wide_adj_nav_tushare.parquet': "6c34a0f19224b2fa28a0b5d1e7baae599b2c0a1370d064bd2d85c0a7cbde641a",
    'data/processed/core13_wide_amount.parquet': "69df0d87c29557616d96840ae118de39a88ea6ceddcdb8446b0e84d15a0681fe",
    'data/processed/core13_wide_close.parquet': "b5ad3b57e34bd6e6533d5381c9c28bd16abb43d22631f63bd90d10d05fbac7a0",
    'data/processed/core13_wide_daily_growth_rate.parquet': "4c2fe2ddb86ed229839bd195ed74e81d27916a6242a371078569e01ecd7f5053",
    'data/processed/core13_wide_high.parquet': "893cd2e3bf74b179c0dfce1d8435edf917f54e137e902d2bdda410f9a360e8e4",
    'data/processed/core13_wide_log_return.parquet': "718d17217860c1f784b5bbba9ceb6839a1ae5324dbe267923d7b82ae90507233",
    'data/processed/core13_wide_low.parquet': "1234dc08dddbf553d1cf5a821bee0d64e2a0745fd5647aa186750d63438daeb3",
    'data/processed/core13_wide_open.parquet': "090f1c64747fad52bdcd3b989351b591d1e23d8fd623e543b1fbc6bde3236512",
    'data/processed/core13_wide_pct_chg.parquet': "54a7ace87eb0c2e2b3e0d08718c8bb1100531b04dbf49ef65c73f62a7180343c",
    'data/processed/core13_wide_pre_close.parquet': "08597cd64fdcdd75def14345c2d04c7c51d0a2b7ac46b01cbf86a00538ee9148",
    'data/processed/core13_wide_turnover_rate.parquet': "dd5421c9eab5e8edbf618b56a1a6523cf7b52914cee56d332d01b2ee0e03d08c",
    'data/processed/core13_wide_unit_nav.parquet': "bc45d9415f5434462a3f4b9587c7c89d89c73cc4474538b23b6bfd867221ab1e",
    'data/processed/core13_wide_vol.parquet': "ba9dd11acba20ea2db3d0bb390528a1514a6ef916d17877ef34f76e367ffd2f5",
    'data/processed/etf_lof_daily_panel.parquet': "7e1695ea15b355bd784a7a95b160215017ffcce1226bec946ce4df39e92cf397",
    'data/processed/wide_amount.parquet': "1d6cffb53d7668728bd0e404012240ae9ec62454d2c994471ff7b87038296dad",
    'data/processed/wide_amount_df.parquet': "1d6cffb53d7668728bd0e404012240ae9ec62454d2c994471ff7b87038296dad",
    'data/processed/wide_close.parquet': "043e82441c8744161ab666528c3ed3d79c33d7791966ec7ac89ae50a84dfd1e7",
    'data/processed/wide_close_df.parquet': "043e82441c8744161ab666528c3ed3d79c33d7791966ec7ac89ae50a84dfd1e7",
    'data/processed/wide_high.parquet': "be36d58c5bb9feaa57ffff9dc4bff571ec3e1a9bd3ab9b3091135af414b5a526",
    'data/processed/wide_high_df.parquet': "be36d58c5bb9feaa57ffff9dc4bff571ec3e1a9bd3ab9b3091135af414b5a526",
    'data/processed/wide_log_return.parquet': "c06f33bd4d005b1115f108debd627871578e1e936bc72c2fb0068d0649ca1f0e",
    'data/processed/wide_log_return_df.parquet': "c06f33bd4d005b1115f108debd627871578e1e936bc72c2fb0068d0649ca1f0e",
    'data/processed/wide_low.parquet': "3f2232974d32c2fec6ff86aad1ea9027cd7c54100f2fc725d42443f112a862f9",
    'data/processed/wide_low_df.parquet': "3f2232974d32c2fec6ff86aad1ea9027cd7c54100f2fc725d42443f112a862f9",
    'data/processed/wide_open.parquet': "c1acad068df25d9099970d7f1924d5c31a35b4a1136875781d94d354c45f3f88",
    'data/processed/wide_open_df.parquet': "c1acad068df25d9099970d7f1924d5c31a35b4a1136875781d94d354c45f3f88",
    'data/processed/wide_pct_chg.parquet': "a8d7605bb5faea775addb92f24410c715032f41eda720f5758e3012ae4b9e17b",
    'data/processed/wide_pct_chg_df.parquet': "a8d7605bb5faea775addb92f24410c715032f41eda720f5758e3012ae4b9e17b",
    'data/processed/wide_pre_close.parquet': "b835032c92702b043b331b9c1f32decd3a99fe389285eb046f37150689f96ab4",
    'data/processed/wide_pre_close_df.parquet': "b835032c92702b043b331b9c1f32decd3a99fe389285eb046f37150689f96ab4",
    'data/processed/wide_turnover_rate.parquet': "dab16392661e60a45d37bb1d3c4bdca932593f51a7e515deea8371b70a119edf",
    'data/processed/wide_turnover_rate_df.parquet': "dab16392661e60a45d37bb1d3c4bdca932593f51a7e515deea8371b70a119edf",
    'data/processed/wide_vol.parquet': "199b6e842dcc17ae57f77a740e974afeea1740b3a97544c6ce2aec2cdf678b28",
    'data/processed/wide_vol_df.parquet': "199b6e842dcc17ae57f77a740e974afeea1740b3a97544c6ce2aec2cdf678b28",
    'data/raw/159912.SZ_daily.parquet': "af5e0249dffd1771f0f29c1f72cdaceed9604d17e4e500d7faf3567c56d76715",
    'data/raw/159915.SZ_daily.parquet': "bdcd1bb7924407bb12d96e9f3d5df19e3b66620b89e45270a90be3bd9c9f53d6",
    'data/raw/159980.SZ_daily.parquet': "c1c532e4882d8031a2df1d959bf946f3211771177ea42abb1f1f70d290328e07",
    'data/raw/159981.SZ_daily.parquet': "233d0679069800a68d11327b197b5dda1f97e78ba2388dc57b23c54213c7331c",
    'data/raw/159985.SZ_daily.parquet': "cfa4993354a464c1ae3d76a024eae8f77aa2ff994db34f5b04954195726a5161",
    'data/raw/161226.SZ_daily.parquet': "c721d5b0e118cec2a1e658cb13ab400f6a12a0552eefd4108f047ad2d84401a5",
    'data/raw/164701.SZ_daily.parquet': "ccb92d755da1fb244822fb4c483ac4d6c6b89e7219ba8d5e7c86f50ad36ddcde",
    'data/raw/501018.SH_daily.parquet': "cca2fc8986e62fa73480287070a0dcf0f789ac412f9bd8738eb8540f849df1d4",
    'data/raw/510050.SH_daily.parquet': "a5a17b20b9737fe3da2131c4a67b94278541467495ebaa80671633dcb51e46a5",
    'data/raw/511010.SH_daily.parquet': "3b77e31e80764899c88742e9e013e47a25709383cd65998c665d0cb28dce67c6",
    'data/raw/511990.SH_daily.parquet': "5c4a2ede6f6ca235077dbcd8567b725c11d666244ba06cc6dae5130202936516",
    'data/raw/512500.SH_daily.parquet': "fa8b00a52dfba67f100b3f3f1cfbc41e64f37a199847bee82647ae24afc0b475",
    'data/raw/513030.SH_daily.parquet': "b90ab99fd00309a0e4ace1427d0ef079eaedd200471bc21b23bde70fe968c290",
    'data/raw/513080.SH_daily.parquet': "3fc724de3899beb028ddd921517a034bae7169b6810f0aca2a58fbbfee8c9591",
    'data/raw/513100.SH_daily.parquet': "57b16f5c3842a22693588dadc5066b5c87781054cf5fde115e0559c2caa151a1",
    'data/raw/513520.SH_daily.parquet': "73645fdca7fb91559b95abe85d2ca0fc7e3e11a206aa7544ddf5e8c56c9f5d4f",
    'data/raw/518880.SH_daily.parquet': "a496ccca10fae40235cc03fb1a73358e3307ce758e3bcad94c97c0afcbe94fcb",
    'data/raw/core13_159915.SZ_daily.parquet': "51b4131b9b674712e4b8b2cb966c172b6dd557847932410ded51b1a4f870cad3",
    'data/raw/core13_159920.SZ_daily.parquet': "ab648403d07b71a1832dc2a8f6236b0a02db9bc0466fec20aed82550807167ae",
    'data/raw/core13_160216.SZ_daily.parquet': "f501aa9b1e7ab93b1853bb1455972ec7fa9b8436cd6918fe8365d0b8a8ea8c46",
    'data/raw/core13_160416.SZ_daily.parquet': "fa527ed003eaaba1198c6c27b092ce03101f4c765dd5b76d1e1b6a4b6f4a2008",
    'data/raw/core13_510050.SH_daily.parquet': "7420f17e5a64df0edb313ac943c5e1ab0b918f67bdcd34036524bc14c0b3a570",
    'data/raw/core13_510300.SH_daily.parquet': "565300aa6aae9852f0af3b37ae093515336ee4627ba26fde1e0c0b44633ccedd",
    'data/raw/core13_510500.SH_daily.parquet': "e8ebc5af64d4e8bebc870c3d41ae6743105998a13f558719f96e094782e86d88",
    'data/raw/core13_510880.SH_daily.parquet': "66386b2b573344fe293f2c9951d8803cc38a23b0243e04c192b05d845b45922d",
    'data/raw/core13_511010.SH_daily.parquet': "28a07dfe3acdeaebe57d106c41148474ab8350c7d9010a573d36fa5229035a14",
    'data/raw/core13_511880.SH_daily.parquet': "6e5d413d0dc5e3d251ac8c8eb2d7d7ce532580f2195717e6408d4f53e719571f",
    'data/raw/core13_513100.SH_daily.parquet': "eb46e91a6af8b98e048942a06f8080ca6fad5106cae86e265c2930fcb69a2775",
    'data/raw/core13_513500.SH_daily.parquet': "54cc001f9aba2fea4e961373665ad0e2614cbe8b7de6a5f9d1129a72b5c1ebf0",
    'data/raw/core13_518880.SH_daily.parquet': "c90cbd44317378c99cbf8e4cceb8102216d041469a34aeb8b247e8ed94666593",
    'data/reports/akshare_etf_lof_download_manifest.json': "c88592eec6acc5619d3a365e66329207517e2a334a04800b7dacd488ff104240",
    'data/reports/all_fund_basic_info_akshare_manifest.json': "8a389f0c2061b13ae1c9a5b451dfaa84cf396e34bfe692a880fbe63406fd3662",
    'data/reports/all_fund_basic_info_akshare_with_establish_dates_manifest.json': "d315b4141d535dba26890dd1664c4013687495caf156d3053b1a2bdec1fea415",
    'data/reports/core13_calendar_loss_summary.json': "17fe9a7588e19007742edf4934744ef5c38fe2617a77a513e4682e05961be3d6",
    'data/reports/core13_data_download_manifest.json': "8fd0b7cab15daa5c492eae297f8d3060905a2934b4b85f28791e8aa2e2fb46e9",
    'data/reports/core13_etf_lof_daily_nav_manifest.json': "2432f94225064c09ae330e6b7a2c719874ab024966a53489e27ec0e6878998b6",
    'data/reports/core13_etf_lof_fund_nav_tushare_manifest.json': "2c2dac91470d414b5df7bfb036368e85f58ac34edcd5286745399a589ef6624c",
    'data/reports/core13_metrics_factory_manifest.json': "d9917a4534f046281f52d144e32a471ba31bce7a2fcaa4f29b0a1a4ee106028a",
    'data/reports/core13_ohlcv_download_manifest.json': "25ba47310afc4db4d311aca31e2cb5e5659e983bb4d518992385694593eb4155",
    'data/reports/etf_lof_basic_info_akshare_manifest.json': "ef94ef92d3d25c5352e939aca8620242b04f9f506a7b633b7b0557573300928c",
    'data/reports/lixinger_fund_info_manifest.json': "5684d237dc878c3c571e0471852f59b328f03157547bc8e6b337f93c256679fe",
    'data/reports/lixinger_fund_info_sample_manifest.json': "bd36c36c5f63367d5071af8ebc4bab50372a35c3030d66d539890737bf500ddc",
    'data/reports/metrics_factory_all_features_manifest.json': "7ec87f76294d45d33d039b82df1e9c3658cdffcbe18ebaf4cfafa31999bf1db5",
    'requirements_data_pipeline.txt': "e41a5323149ac2fd4b1e859e1d0d9057d1066960eeaed27094e9366e2179a8cb",
    'scripts/rebuild_etf_lof_data.py': "66df9826f05bd303656c9777d9a253354fafcf9df9b266adcd0d72a6cd18aaae",
    'scripts/run_pyfinance_metrics_factory.py': "282824c55dcf99bda9a8be58090604b6e5de63b42c6213caf9e845f5c41b0158",
    'vendor/pyfinance/AutoFactorCreator/A01_DataPrepare.py': "81dcc7c7ce8e0ea8a02ff3acee74d0ec748e3fe67ce5de6dc7fe3fb373782c03",
    'vendor/pyfinance/AutoFactorCreator/A02_OperatorLibrary.py': "3768a7d9d98d8835ffc0d53b5c2ebf3264664950e6a9a89d0637851db37da9e8",
    'vendor/pyfinance/AutoFactorCreator/A03_FinancialAgent.py': "464158344ec4e85af49a66051e3687fab0451840d40eef67bf627de16494b68f",
    'vendor/pyfinance/AutoFactorCreator/A04_AnalysisAgent.py': "270b3859f4761f136fae7717138aef1f0d7037339c5ba8bddde8cb81fbc15f47",
    'vendor/pyfinance/AutoFactorCreator/A05_PythonAgent.py': "91eb8c2ca69d04ad21983f39b41664614a9b4957748e3f0f5c46fc415eb9c504",
    'vendor/pyfinance/AutoFactorCreator/A06_CalFactors.py': "7b74d04a81c23e64b4b50a5359a4d5c3abd17ce899bbaac29a89b474f97597cc",
    'vendor/pyfinance/AutoFactorCreator/A07_Scheduler.py': "b7e8ce565c34b6ec0baa1d12ef0f5bb145c4edf625e87ebe6e9405d5312e4147",
    'vendor/pyfinance/AutoFactorCreator/B01_OperatorTest.py': "876e7b83db8c314c3e61a2ee7ed472b6ac894478b816268509183c78accec9f7",
    'vendor/pyfinance/AutoFactorCreator/B02_AgentTools.py': "e0e3f0316860779ee9b8287ee08c8f03fe9d5fd668623083cec3ea40d139a99b",
    'vendor/pyfinance/AutoFactorCreator/DesignDoc.md': "3f5e87b6a2d36cd3ee7f9f5352d28c392188fd3f7638fc006aa29e9d9b389e56",
    'vendor/pyfinance/AutoFactorCreator/config.json': "04385831479eacc9b02344dd6d248a98a9bbb1e14f7fc9f166f4bdd8c65a497a",
    'vendor/pyfinance/GetData/akshare_get_INDEX_data.py': "8258a630cfbba3f426d6ec1b998010edd8d9e56655f3144316bcd990874ffc8f",
    'vendor/pyfinance/GetData/data_prepare.py': "ac232cb8a5927ff205f60c2efe2cced24018bfb3b95d14faf5d4a75185d8f11c",
    'vendor/pyfinance/GetData/tushare_get_ETF_data.py': "a480155f2617ff2c251a42809b4a228d5858fd429d9236a5a444cbbb41ee652f",
    'vendor/pyfinance/MetricsFactory/metrics_cal_config.py': "c997c4c650d7e4f5a533e86824d480bc3938162a98a84896a6352e729d149dc5",
    'vendor/pyfinance/MetricsFactory/metrics_factory.py': "1d48920b2822a10b472a43b03f27eec82a1926e3a59fc465c249ee0032eb55ab",
    'vendor/pyfinance/MetricsFactory/period_metrics_cal.py': "24841e483453c01f2633f9f771e61c8fd7c9dd7d8ef1427f30b347eec533033d",
    'vendor/pyfinance/MetricsFactory/rolling_metrics_cal.py': "ca58322c60ba0656af097ca8fea55466d6577b65f8f8924ecceee5ae40e03dc8",
    'vendor/pyfinance/MetricsFactory/指标说明.xlsx': "a12bf749232df3eb686261ca52bc9f522441bace0cec1810445f08efdf27133d",
    'vendor/pyfinance/README.md': "72bde9fc61aa94132a0b62d7c83174d45f4c85781ac9b4ccb62cf55b801e69c9",
    'vendor/pyfinance/legacy_rl_scripts/PPO_TRADE.py': "8c1d54d4c59fe1e38c6164a1c69a49a830cd8464d639a604c310c95e49a15ad8",
    'vendor/pyfinance/legacy_rl_scripts/test_DQN_GPU_noisy.py': "cc4785b96924372d1fd0967864ccb482984b962c3dedfb2479b4bb7ef20f8a89",
    'vendor/pyfinance/legacy_rl_scripts/test_EIIE.py': "945399ed46e85c40b8cc9acf4b795979fcdbc9cee8057a59ee8259ef3f312a30",
    'vendor/pyfinance/legacy_rl_scripts/test_PPO.py': "4395a48f0606e568691a910dccbb76e711d2e471f2374ef62f7e94a68702a74d",
    'vendor/pyfinance/legacy_rl_scripts/test_PPO_CNN.py': "f08a826f00193cf325a07f7b2f375c30b7bcd6fd66aa7563d3743b64c9b3ba3f",
    'vendor/pyfinance/legacy_rl_scripts/test_PPO_CNN_integrate.py': "a4c55b9a7e9f73866bba3a5d4ad4d53176204b51823198ce8f07f89ea0f57a8d",
    'vendor/pyfinance/main_cal_metrics.py': "981e3952fadb5ebd1b659974730bed07ebaffbda29f441ffe61b692b3c3dc77b",
    'vendor/pyfinance/main_get_data.py': "16badf5507c6d6812a3717a8be63238799ddff86d4697cb414553cb200a19975",
    'vendor/pyfinance/set_tushare.py': "147952ee42105e77d1e2e5ea39d4733381ade312d33a7674665d7be53fbc9e4f",
}


def test_scope_guard_freezes_non_platform_files():
    guarded_paths = set(_discover_guarded_paths())
    expected_paths = set(EXPECTED_GUARDED_FILE_SHA256)

    assert sorted(guarded_paths - expected_paths) == []
    assert sorted(expected_paths - guarded_paths) == []

    actual_hashes = {path: _sha256(PROJECT_ROOT / path) for path in EXPECTED_GUARDED_FILE_SHA256}
    mismatched = {
        path: actual_sha256
        for path, actual_sha256 in actual_hashes.items()
        if actual_sha256 != EXPECTED_GUARDED_FILE_SHA256[path]
    }
    assert mismatched == {}


def test_pytest_discovery_boundary():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]

    assert pytest_options["testpaths"] == ["tests"]
    assert pytest_options["pythonpath"] == ["."]

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    collected = [line for line in result.stdout.splitlines() if "::" in line]
    assert collected
    assert all(line.startswith("tests/") for line in collected)
    assert ".venv/" not in result.stdout
    assert "vendor/pyfinance/legacy_rl_scripts" not in result.stdout


def _discover_guarded_paths() -> list[str]:
    paths = {path for path in STATIC_GUARDED_FILES if (PROJECT_ROOT / path).is_file()}
    paths.update(_relative_files(PROJECT_ROOT / "vendor" / "pyfinance"))
    paths.update(_relative_files(PROJECT_ROOT / "data", suffix=".parquet"))
    paths.update(_relative_files(PROJECT_ROOT / "data" / "reports", suffix=".json", recursive=False))
    return sorted(paths)


def _relative_files(root: Path, suffix: str | None = None, recursive: bool = True) -> set[str]:
    if not root.exists():
        return set()
    iterator = root.rglob("*") if recursive else root.glob("*")
    return {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in iterator
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.endswith((".pyc", ".pyo"))
        and (suffix is None or path.suffix == suffix)
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
