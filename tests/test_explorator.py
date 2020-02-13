"""Unit tests for lambdata_jcslambda.df_utils module"""

import unittest

from lambdata_jcslambda.df_utils import pd, np
from lambdata_jcslambda.explorator import Explorator

RANDOM_SEED = 13


class Test_Explorator(unittest.TestCase):
    """Test the lambdata_jcslambda.explorator.Explorator class."""

    def setUp(self):
        """Initialize testing data.

        Defines a random seed for consistency across tests.

        Creates a dataframe for use in tests.
        """
        self.random_state = RANDOM_SEED
        np.random.seed(RANDOM_SEED)
        dates = pd.date_range(
            start='2017-01-01',
            end='2018-12-31',
            freq='W-TUE'
        )
        small_numbers = np.random.randint(0, 25, len(dates))
        large_numbers = np.random.randint(100, 10000, len(dates))
        wld = ['win', 'lose', 'draw']
        results = np.random.choice(wld, len(dates), True)
        data = {
            'date': dates,
            'small_num': small_numbers,
            'large_num': large_numbers,
            'result': results,
        }
        self.xdf = Explorator(pd.DataFrame(data), self.random_state)

    def tearDown(self):
        """Clean up after each test."""
        del self.xdf
        del self.random_state

    def test_tvt_split(self):
        """Test tvt_split"""
        original_shape = self.xdf.df.shape
        # test train, val, and test attributes do not exist
        # prior to calling tvt_split
        with self.assertRaises(AttributeError):
            self.xdf.train is None
        with self.assertRaises(AttributeError):
            self.xdf.val is None
        with self.assertRaises(AttributeError):
            self.xdf.test is None
        self.xdf.tvt_split(
            target='result',
        )
        train = self.xdf.train
        val = self.xdf.val
        test = self.xdf.test
        # test if original dataframe was altered
        self.assertEqual(original_shape, self.xdf.df.shape)
        # test return values are dataframes
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(val, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
        # test y shape of returned dataframes
        self.assertEqual(train.shape[1], self.xdf.df.shape[1])
        self.assertEqual(val.shape[1], self.xdf.df.shape[1])
        self.assertEqual(test.shape[1], self.xdf.df.shape[1])
        # test x shape of returned dataframes
        tvt_x_shape_sum = train.shape[0] + val.shape[0] + test.shape[0]
        self.assertEqual(tvt_x_shape_sum, self.xdf.df.shape[0])

    def test_expand_date_parts(self):
        """Test expand_date_parts"""
        pass

    def test_describe(self):
        """Test describe"""
        pass

    def test_value_counts(self):
        """Test value_counts"""
        pass

    def test_barplot_feat_by_target_eq_class(self):
        """Test barplot_feat_by_target_eq_class"""
        pass

    def test_barplots_low_card_feat_by_target_eq_class(self):
        """Test barplots_low_card_feat_by_target_eq_class"""
        pass


if __name__ == '__main__':
    unittest.main()
