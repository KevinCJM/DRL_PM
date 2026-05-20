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
    "data/metrics_factory/10d.parquet": "067dcec768a75692acac7957267568dd8034d2edcb8656e4e1746842ccb63115",
    "data/metrics_factory/12m.parquet": "1df6c8ab4fdd568ab4e4614210a957cae799f89455813ae43b3b3b3ffd40e8f3",
    "data/metrics_factory/15d.parquet": "b83057278e3d4c2811c0550a691a3786a85ff3677fc8414cd22319a022331f47",
    "data/metrics_factory/20d.parquet": "f43145569df15456ba66c861b5d1d093b2a85d550c8ee2c51975c4179289482e",
    "data/metrics_factory/25d.parquet": "753b93ae4d06eca54b2afd6e4eae1aa6e2606153fc3f7dbac3777dfc8ddc4d6a",
    "data/metrics_factory/2d.parquet": "0fa3664562e78cc5ec5dd5b94341418476462867104312ca825fa214a774d1f9",
    "data/metrics_factory/2y.parquet": "1814cc06efc686d3810b02c89fca17bfec43cb1a4902366c28839df1ee331a18",
    "data/metrics_factory/3d.parquet": "d918ca289f2522e05ab662c642e4befb83f16c015abfac609ecf27cac36ab4a1",
    "data/metrics_factory/50d.parquet": "948915b7c8f3ec13c57bcc315c6d8bc7c896a6dea6600270e8cf4ec707fdce09",
    "data/metrics_factory/5d.parquet": "6c74637cb53fc0432bf067508291e3a0d433fc7409a1787716d5135c7f07dfd0",
    "data/metrics_factory/5m.parquet": "df6afbfaf9f93ed22cc68ddb716a87fe66f5549221f4b5241f37b00bdb30e5c8",
    "data/metrics_factory/6d.parquet": "9f8e6a35d08a223c1b5348c1ce2a453814ce7f58c9228a3ec887ae8e52066554",
    "data/metrics_factory/6m.parquet": "3ec5f7787968fc092fd524dc92ecfa4d7a7f77f984ce65ab13e1603ec4eae498",
    "data/metrics_factory/75d.parquet": "ee3853a2a9d62cdc84f048100f1a1b60f76fee51a0db43a41459be0d1e490d1d",
    "data/metrics_factory/7d.parquet": "fc49a9036b1828c67bd23fc41290fb5d6a4a389088a15b5d1feded5db89ac5a8",
    "data/metrics_factory/9m.parquet": "a2d3fe2f4ca5d7480360ba51826216ebcbddbb48c31a3de9c2782bcbb64b11c9",
    "data/metrics_factory/all_metrics_features.parquet": "f665c1b974d6a8e0e4fb4cf6184c189fd680f4bd31f0c420a6e53b47cece5665",
    "data/metrics_factory/rolling_metrics.parquet": "367b2d120c6268b927329dcf5285d59f3b5aacc120e4658ce6165ef5305b6f3f",
    "data/processed/etf_lof_daily_panel.parquet": "7e1695ea15b355bd784a7a95b160215017ffcce1226bec946ce4df39e92cf397",
    "data/processed/wide_amount.parquet": "1d6cffb53d7668728bd0e404012240ae9ec62454d2c994471ff7b87038296dad",
    "data/processed/wide_amount_df.parquet": "1d6cffb53d7668728bd0e404012240ae9ec62454d2c994471ff7b87038296dad",
    "data/processed/wide_close.parquet": "043e82441c8744161ab666528c3ed3d79c33d7791966ec7ac89ae50a84dfd1e7",
    "data/processed/wide_close_df.parquet": "043e82441c8744161ab666528c3ed3d79c33d7791966ec7ac89ae50a84dfd1e7",
    "data/processed/wide_high.parquet": "be36d58c5bb9feaa57ffff9dc4bff571ec3e1a9bd3ab9b3091135af414b5a526",
    "data/processed/wide_high_df.parquet": "be36d58c5bb9feaa57ffff9dc4bff571ec3e1a9bd3ab9b3091135af414b5a526",
    "data/processed/wide_log_return.parquet": "c06f33bd4d005b1115f108debd627871578e1e936bc72c2fb0068d0649ca1f0e",
    "data/processed/wide_log_return_df.parquet": "c06f33bd4d005b1115f108debd627871578e1e936bc72c2fb0068d0649ca1f0e",
    "data/processed/wide_low.parquet": "3f2232974d32c2fec6ff86aad1ea9027cd7c54100f2fc725d42443f112a862f9",
    "data/processed/wide_low_df.parquet": "3f2232974d32c2fec6ff86aad1ea9027cd7c54100f2fc725d42443f112a862f9",
    "data/processed/wide_open.parquet": "c1acad068df25d9099970d7f1924d5c31a35b4a1136875781d94d354c45f3f88",
    "data/processed/wide_open_df.parquet": "c1acad068df25d9099970d7f1924d5c31a35b4a1136875781d94d354c45f3f88",
    "data/processed/wide_pct_chg.parquet": "a8d7605bb5faea775addb92f24410c715032f41eda720f5758e3012ae4b9e17b",
    "data/processed/wide_pct_chg_df.parquet": "a8d7605bb5faea775addb92f24410c715032f41eda720f5758e3012ae4b9e17b",
    "data/processed/wide_pre_close.parquet": "b835032c92702b043b331b9c1f32decd3a99fe389285eb046f37150689f96ab4",
    "data/processed/wide_pre_close_df.parquet": "b835032c92702b043b331b9c1f32decd3a99fe389285eb046f37150689f96ab4",
    "data/processed/wide_turnover_rate.parquet": "dab16392661e60a45d37bb1d3c4bdca932593f51a7e515deea8371b70a119edf",
    "data/processed/wide_turnover_rate_df.parquet": "dab16392661e60a45d37bb1d3c4bdca932593f51a7e515deea8371b70a119edf",
    "data/processed/wide_vol.parquet": "199b6e842dcc17ae57f77a740e974afeea1740b3a97544c6ce2aec2cdf678b28",
    "data/processed/wide_vol_df.parquet": "199b6e842dcc17ae57f77a740e974afeea1740b3a97544c6ce2aec2cdf678b28",
    "data/raw/159912.SZ_daily.parquet": "af5e0249dffd1771f0f29c1f72cdaceed9604d17e4e500d7faf3567c56d76715",
    "data/raw/159915.SZ_daily.parquet": "bdcd1bb7924407bb12d96e9f3d5df19e3b66620b89e45270a90be3bd9c9f53d6",
    "data/raw/159980.SZ_daily.parquet": "c1c532e4882d8031a2df1d959bf946f3211771177ea42abb1f1f70d290328e07",
    "data/raw/159981.SZ_daily.parquet": "233d0679069800a68d11327b197b5dda1f97e78ba2388dc57b23c54213c7331c",
    "data/raw/159985.SZ_daily.parquet": "cfa4993354a464c1ae3d76a024eae8f77aa2ff994db34f5b04954195726a5161",
    "data/raw/161226.SZ_daily.parquet": "c721d5b0e118cec2a1e658cb13ab400f6a12a0552eefd4108f047ad2d84401a5",
    "data/raw/164701.SZ_daily.parquet": "ccb92d755da1fb244822fb4c483ac4d6c6b89e7219ba8d5e7c86f50ad36ddcde",
    "data/raw/501018.SH_daily.parquet": "cca2fc8986e62fa73480287070a0dcf0f789ac412f9bd8738eb8540f849df1d4",
    "data/raw/510050.SH_daily.parquet": "a5a17b20b9737fe3da2131c4a67b94278541467495ebaa80671633dcb51e46a5",
    "data/raw/511010.SH_daily.parquet": "3b77e31e80764899c88742e9e013e47a25709383cd65998c665d0cb28dce67c6",
    "data/raw/511990.SH_daily.parquet": "5c4a2ede6f6ca235077dbcd8567b725c11d666244ba06cc6dae5130202936516",
    "data/raw/512500.SH_daily.parquet": "fa8b00a52dfba67f100b3f3f1cfbc41e64f37a199847bee82647ae24afc0b475",
    "data/raw/513030.SH_daily.parquet": "b90ab99fd00309a0e4ace1427d0ef079eaedd200471bc21b23bde70fe968c290",
    "data/raw/513080.SH_daily.parquet": "3fc724de3899beb028ddd921517a034bae7169b6810f0aca2a58fbbfee8c9591",
    "data/raw/513100.SH_daily.parquet": "57b16f5c3842a22693588dadc5066b5c87781054cf5fde115e0559c2caa151a1",
    "data/raw/513520.SH_daily.parquet": "73645fdca7fb91559b95abe85d2ca0fc7e3e11a206aa7544ddf5e8c56c9f5d4f",
    "data/raw/518880.SH_daily.parquet": "a496ccca10fae40235cc03fb1a73358e3307ce758e3bcad94c97c0afcbe94fcb",
    "data/reports/akshare_etf_lof_download_manifest.json": "c88592eec6acc5619d3a365e66329207517e2a334a04800b7dacd488ff104240",
    "data/reports/metrics_factory_all_features_manifest.json": "7ec87f76294d45d33d039b82df1e9c3658cdffcbe18ebaf4cfafa31999bf1db5",
    "requirements_data_pipeline.txt": "d61b118d6534828ab9383fe1880e73e16872749b03f934671ad8be3d382b3dd6",
    "scripts/rebuild_etf_lof_data.py": "d50e3033529ca1e8fa10ce42aa26003cab06cee4783ca2890360388067700366",
    "scripts/run_pyfinance_metrics_factory.py": "282824c55dcf99bda9a8be58090604b6e5de63b42c6213caf9e845f5c41b0158",
    "vendor/pyfinance/AutoFactorCreator/A01_DataPrepare.py": "81dcc7c7ce8e0ea8a02ff3acee74d0ec748e3fe67ce5de6dc7fe3fb373782c03",
    "vendor/pyfinance/AutoFactorCreator/A02_OperatorLibrary.py": "3768a7d9d98d8835ffc0d53b5c2ebf3264664950e6a9a89d0637851db37da9e8",
    "vendor/pyfinance/AutoFactorCreator/A03_FinancialAgent.py": "464158344ec4e85af49a66051e3687fab0451840d40eef67bf627de16494b68f",
    "vendor/pyfinance/AutoFactorCreator/A04_AnalysisAgent.py": "270b3859f4761f136fae7717138aef1f0d7037339c5ba8bddde8cb81fbc15f47",
    "vendor/pyfinance/AutoFactorCreator/A05_PythonAgent.py": "91eb8c2ca69d04ad21983f39b41664614a9b4957748e3f0f5c46fc415eb9c504",
    "vendor/pyfinance/AutoFactorCreator/A06_CalFactors.py": "7b74d04a81c23e64b4b50a5359a4d5c3abd17ce899bbaac29a89b474f97597cc",
    "vendor/pyfinance/AutoFactorCreator/A07_Scheduler.py": "b7e8ce565c34b6ec0baa1d12ef0f5bb145c4edf625e87ebe6e9405d5312e4147",
    "vendor/pyfinance/AutoFactorCreator/B01_OperatorTest.py": "876e7b83db8c314c3e61a2ee7ed472b6ac894478b816268509183c78accec9f7",
    "vendor/pyfinance/AutoFactorCreator/B02_AgentTools.py": "e0e3f0316860779ee9b8287ee08c8f03fe9d5fd668623083cec3ea40d139a99b",
    "vendor/pyfinance/AutoFactorCreator/DesignDoc.md": "3f5e87b6a2d36cd3ee7f9f5352d28c392188fd3f7638fc006aa29e9d9b389e56",
    "vendor/pyfinance/AutoFactorCreator/config.json": "04385831479eacc9b02344dd6d248a98a9bbb1e14f7fc9f166f4bdd8c65a497a",
    "vendor/pyfinance/GetData/akshare_get_INDEX_data.py": "8258a630cfbba3f426d6ec1b998010edd8d9e56655f3144316bcd990874ffc8f",
    "vendor/pyfinance/GetData/data_prepare.py": "ac232cb8a5927ff205f60c2efe2cced24018bfb3b95d14faf5d4a75185d8f11c",
    "vendor/pyfinance/GetData/tushare_get_ETF_data.py": "a480155f2617ff2c251a42809b4a228d5858fd429d9236a5a444cbbb41ee652f",
    "vendor/pyfinance/MetricsFactory/metrics_cal_config.py": "c997c4c650d7e4f5a533e86824d480bc3938162a98a84896a6352e729d149dc5",
    "vendor/pyfinance/MetricsFactory/metrics_factory.py": "1d48920b2822a10b472a43b03f27eec82a1926e3a59fc465c249ee0032eb55ab",
    "vendor/pyfinance/MetricsFactory/period_metrics_cal.py": "24841e483453c01f2633f9f771e61c8fd7c9dd7d8ef1427f30b347eec533033d",
    "vendor/pyfinance/MetricsFactory/rolling_metrics_cal.py": "ca58322c60ba0656af097ca8fea55466d6577b65f8f8924ecceee5ae40e03dc8",
    "vendor/pyfinance/MetricsFactory/指标说明.xlsx": "a12bf749232df3eb686261ca52bc9f522441bace0cec1810445f08efdf27133d",
    "vendor/pyfinance/README.md": "72bde9fc61aa94132a0b62d7c83174d45f4c85781ac9b4ccb62cf55b801e69c9",
    "vendor/pyfinance/legacy_rl_scripts/PPO_TRADE.py": "8c1d54d4c59fe1e38c6164a1c69a49a830cd8464d639a604c310c95e49a15ad8",
    "vendor/pyfinance/legacy_rl_scripts/test_DQN_GPU_noisy.py": "cc4785b96924372d1fd0967864ccb482984b962c3dedfb2479b4bb7ef20f8a89",
    "vendor/pyfinance/legacy_rl_scripts/test_EIIE.py": "945399ed46e85c40b8cc9acf4b795979fcdbc9cee8057a59ee8259ef3f312a30",
    "vendor/pyfinance/legacy_rl_scripts/test_PPO.py": "4395a48f0606e568691a910dccbb76e711d2e471f2374ef62f7e94a68702a74d",
    "vendor/pyfinance/legacy_rl_scripts/test_PPO_CNN.py": "f08a826f00193cf325a07f7b2f375c30b7bcd6fd66aa7563d3743b64c9b3ba3f",
    "vendor/pyfinance/legacy_rl_scripts/test_PPO_CNN_integrate.py": "a4c55b9a7e9f73866bba3a5d4ad4d53176204b51823198ce8f07f89ea0f57a8d",
    "vendor/pyfinance/main_cal_metrics.py": "981e3952fadb5ebd1b659974730bed07ebaffbda29f441ffe61b692b3c3dc77b",
    "vendor/pyfinance/main_get_data.py": "16badf5507c6d6812a3717a8be63238799ddff86d4697cb414553cb200a19975",
    "vendor/pyfinance/set_tushare.py": "147952ee42105e77d1e2e5ea39d4733381ade312d33a7674665d7be53fbc9e4f",
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
