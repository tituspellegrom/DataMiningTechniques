import math
import pandas as pd


class MetronAtK(object):
    """Object-Oriented class to calculate metrics for Top-K list"""

    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """

        assert isinstance(subjects, list)

        # Get test data
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        # Get negative data
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]

        # the golden set with only test data
        test = pd.DataFrame({'user': test_users, 'test_item': test_items, 'test_score': test_scores})

        # the full set of test and negative data
        full = pd.DataFrame(
            {'user': neg_users + test_users, 'item': neg_items + test_items, 'score': neg_scores + test_scores})
        full = pd.merge(full, test, on=['user'], how='left')

        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)

        self._subjects = full

    def cal_hit_ratio(self):
        """Function to calculate Hit Ratio @ top_K"""

        full, top_k = self._subjects, self._top_k

        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]

        # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):
        """Function to calculate Normalized Discounted Cumulative Gain @ top_K"""

        full, top_k = self._subjects, self._top_k

        top_k = full[full['rank'] <= top_k]

        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1

        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
