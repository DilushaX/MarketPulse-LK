import os
import sys
import json
import unittest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestMarketPulseProject(unittest.TestCase):
    
    # -------------------------------------------------------------
    # Test 1: Dependencies and Environment
    # -------------------------------------------------------------
    def test_01_dependencies_importable(self):
        """Verify all critical scientific & econometric packages import cleanly."""
        modules = [
            'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
            'statsmodels', 'arch', 'sklearn', 'nbformat', 'nbconvert'
        ]
        for mod in modules:
            with self.subTest(module=mod):
                imported = __import__(mod)
                self.assertIsNotNone(imported)
                self.assertTrue(hasattr(imported, '__version__'))

    # -------------------------------------------------------------
    # Test 2: Raw Datasets Integrity
    # -------------------------------------------------------------
    def test_02_raw_datasets(self):
        """Verify raw datasets exist, contain required columns and reasonable row counts."""
        equities = ['JKH', 'HAYL', 'VONE']
        raw_dir = os.path.join(BASE_DIR, 'data', 'raw')
        
        for eq in equities:
            with self.subTest(equity=eq):
                path = os.path.join(raw_dir, f"{eq}_2014_2026.csv")
                self.assertTrue(os.path.exists(path), f"Missing raw file: {path}")
                df = pd.read_csv(path)
                self.assertGreater(len(df), 2000, f"Raw data for {eq} has fewer than 2000 rows")
                expected_cols = {'Date', 'Open', 'High', 'Low', 'Close'}
                self.assertTrue(expected_cols.issubset(set(df.columns)), f"Missing columns in {eq}: {expected_cols - set(df.columns)}")

    # -------------------------------------------------------------
    # Test 3: Processed Cleaned Datasets Integrity
    # -------------------------------------------------------------
    def test_03_processed_datasets(self):
        """Verify cleaned datasets conform to Part 03 preprocessing standards."""
        equities = ['JKH', 'HAYL', 'VONE']
        proc_dir = os.path.join(BASE_DIR, 'data', 'processed')
        
        for eq in equities:
            with self.subTest(equity=eq):
                path = os.path.join(proc_dir, f"{eq}_cleaned.csv")
                self.assertTrue(os.path.exists(path), f"Missing cleaned file: {path}")
                df = pd.read_csv(path)
                
                # Check required columns
                required = ['Date', 'Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume', 'Change_%', 'Log_Return']
                for col in required:
                    self.assertIn(col, df.columns, f"Cleaned {eq} missing column: {col}")
                
                # Chronological monotonic ordering
                dates = pd.to_datetime(df['Date'])
                self.assertTrue(dates.is_monotonic_increasing, f"Dates for {eq} are not monotonically increasing")
                self.assertEqual(dates.duplicated().sum(), 0, f"Duplicate dates detected in {eq}")
                
                # Ensure no NaNs in price columns
                for col in ['Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume']:
                    self.assertEqual(df[col].isnull().sum(), 0, f"NaNs detected in {eq} column {col}")
                
                # Ensure prices are strictly positive
                self.assertTrue((df['Adjusted_Close'] > 0).all(), f"Non-positive prices in {eq}")

    # -------------------------------------------------------------
    # Test 4: Output Tables Integrity
    # -------------------------------------------------------------
    def test_04_output_tables(self):
        """Verify all analytical output tables exist and contain valid data."""
        tab_dir = os.path.join(BASE_DIR, 'outputs', 'tables')
        expected_tables = [
            'adf_test_results.csv',
            'kpss_test_results.csv',
            'stationarity_summary.csv',
            'ljung_box_results.csv',
            'normality_test_summary.csv',
            'arch_lm_results.csv',
            'structural_break_results.csv',
            'asymmetric_effects_results.csv',
            'ols_regression_results.csv',
            'descriptive_statistics.csv',
            'company_comparison.csv',
            'pearson_correlation.csv',
            'spearman_correlation.csv',
            'volatility_summary.csv',
            'return_statistics.csv',
            'return_distribution_statistics.csv'
        ]
        
        for tab in expected_tables:
            with self.subTest(table=tab):
                p = os.path.join(tab_dir, tab)
                self.assertTrue(os.path.exists(p), f"Missing table: {p}")
                self.assertGreater(os.path.getsize(p), 50, f"Table {tab} is empty or too small")
                df = pd.read_csv(p)
                self.assertGreater(len(df), 0, f"Table {tab} contains no rows")

    # -------------------------------------------------------------
    # Test 5: Output Figures Integrity
    # -------------------------------------------------------------
    def test_05_output_figures(self):
        """Verify all generated visualization figures exist and are non-empty images."""
        fig_dir = os.path.join(BASE_DIR, 'outputs', 'figures')
        adv_dir = os.path.join(fig_dir, 'advanced_statistics')
        
        adv_figures = [
            'acf_returns.png',
            'pacf_returns.png',
            'acf_squared_returns.png',
            'pacf_squared_returns.png',
            'acf_absolute_returns.png',
            'cusum_stability_plots.png',
            'zivot_andrews_structural_breaks.png',
            'regression_market_models.png',
            'regression_volume_volatility.png'
        ]
        
        for fig in adv_figures:
            with self.subTest(figure=fig):
                p = os.path.join(adv_dir, fig)
                self.assertTrue(os.path.exists(p), f"Missing figure: {p}")
                self.assertGreater(os.path.getsize(p), 10000, f"Figure {fig} is unusually small (<10KB)")

    # -------------------------------------------------------------
    # Test 6: Econometric Sanity Checks
    # -------------------------------------------------------------
    def test_06_econometric_sanity_results(self):
        """Verify econometric test results match theoretical financial stylised facts."""
        tab_dir = os.path.join(BASE_DIR, 'outputs', 'tables')
        
        # 1. Stationarity
        df_stat = pd.read_csv(os.path.join(tab_dir, 'stationarity_summary.csv'))
        self.assertEqual(len(df_stat), 9)
        for _, row in df_stat.iterrows():
            if 'Price' in row['Variable']:
                self.assertGreater(row['ADF_pvalue'], 0.05, f"{row['Company']} price is unexpectedly ADF stationary")
                self.assertLess(row['KPSS_pvalue'], 0.05, f"{row['Company']} price is unexpectedly KPSS stationary")
            elif 'Return' in row['Variable']:
                self.assertLess(row['ADF_pvalue'], 0.05, f"{row['Company']} return is not stationary according to ADF")
                self.assertGreaterEqual(row['KPSS_pvalue'], 0.05, f"{row['Company']} return KPSS rejects stationarity")
        
        # 2. ARCH-LM Effects
        df_arch = pd.read_csv(os.path.join(tab_dir, 'arch_lm_results.csv'))
        self.assertGreaterEqual(len(df_arch), 6)
        for _, row in df_arch.iterrows():
            self.assertLess(row['LM_p_value'], 0.05, f"ARCH effect not detected in {row['Company']} at lag {row['Lag']}")
            
        # 3. Normality Rejection
        df_norm = pd.read_csv(os.path.join(tab_dir, 'normality_test_summary.csv'))
        self.assertEqual(len(df_norm), 3)
        for _, row in df_norm.iterrows():
            self.assertLess(row['p_value'], 0.05, f"Normal distribution not rejected for {row['Company']}")
            self.assertGreater(row['Full_Kurtosis'], 10.0, f"Kurtosis unexpectedly low for {row['Company']}")
            
        # 4. OLS Regressions
        df_ols = pd.read_csv(os.path.join(tab_dir, 'ols_regression_results.csv'))
        self.assertEqual(len(df_ols), 5)
        for _, row in df_ols.iterrows():
            self.assertGreater(row['Beta (Slope)'], 0.0, f"Negative slope in {row['Dependent Var (Y)']}")
            self.assertLess(row['Beta p-val'], 0.05, f"Insignificant slope in {row['Dependent Var (Y)']}")

    # -------------------------------------------------------------
    # Test 7: Notebooks Valid Structure and Execution Quality
    # -------------------------------------------------------------
    def test_07_notebooks_execution_quality(self):
        """Verify that notebooks parse correctly and have no unhandled execution errors."""
        import nbformat
        notebooks = [
            '01_data_cleaning.ipynb',
            '02_exploratory_analysis.ipynb',
            '03_advanced_statistical_analysis.ipynb'
        ]
        
        for nb_file in notebooks:
            with self.subTest(notebook=nb_file):
                p = os.path.join(BASE_DIR, 'notebooks', nb_file)
                self.assertTrue(os.path.exists(p), f"Missing notebook: {p}")
                nb = nbformat.read(p, as_version=4)
                
                # Check for errors in cell outputs
                for idx, cell in enumerate(nb.cells):
                    if cell.cell_type == 'code':
                        for out in cell.get('outputs', []):
                            self.assertNotEqual(
                                out.get('output_type'), 'error',
                                f"Error output in {nb_file} cell {idx}: {out.get('ename')}: {out.get('evalue')}"
                            )

if __name__ == '__main__':
    unittest.main(verbosity=2)
