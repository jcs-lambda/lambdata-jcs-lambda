"""Unit tests for lambdata_jcslambda.df_utils module"""

import unittest

from lambdata_jcslambda.df_utils import pd, np, tvt_split, expand_date_parts
from lambdata_jcslambda.df_utils import describe, value_counts
from lambdata_jcslambda.df_utils import barplot_feat_by_target_eq_class
from lambdata_jcslambda.df_utils import \
    barplots_low_card_feat_by_target_eq_class

RANDOM_SEED = 13


class Test_df_utils(unittest.TestCase):
    """Test functions in the lambdata_jcslambda.df_utils module."""

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
        self.df = pd.DataFrame(data)

    def tearDown(self):
        """Cleanup after each test."""
        del self.random_state
        del self.df

    def test_tvt_split(self):
        """Test tvt_split"""
        original_shape = self.df.shape
        train, val, test = tvt_split(
            df=self.df,
            target='result',
            random_state=self.random_state
        )
        # test if original dataframe was altered
        self.assertEqual(original_shape, self.df.shape)
        # test return values are dataframes
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(val, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
        # test y shape of returned dataframes
        self.assertEqual(train.shape[1], self.df.shape[1])
        self.assertEqual(val.shape[1], self.df.shape[1])
        self.assertEqual(test.shape[1], self.df.shape[1])
        # test x shape of returned dataframes
        tvt_x_shape_sum = train.shape[0] + val.shape[0] + test.shape[0]
        self.assertEqual(tvt_x_shape_sum, self.df.shape[0])

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
