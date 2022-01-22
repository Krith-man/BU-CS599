import pytest
from assignment_4 import *

# Returns a True/False in a response to induct a tuple in the selection operator
def select_predicate_friends(t):
    if int(t.tuple[0]) == 10:
        return True
    return False


# Returns a True/False in a response to induct a tuple in the selection operator
def select_predicate_movie_ratings(t):
    if int(t.tuple[1]) == 3:
        return True
    return False


class TestClass:

    # Test the bottom-up architecture using a simple test dataset to check if works properly.
    def test_ray(self):
        # Sink operator instantiation
        sink_handler = Sink.remote()
        # Project operator instantiation
        projection_handler = Project.remote(input=None, fields_to_keep=[0], batch_size=1, track_prov=False,
                                            propagate_prov=False)
        projection_handler.set_next_handler.remote(sink_handler)
        # TopK operator instantiation
        topK_handler = TopK.remote(input=None, k=0, track_prov=False, propagate_prov=False)
        topK_handler.set_next_handler.remote(projection_handler)
        # OrderBy operator instantiation
        order_handler = OrderBy.remote(input=None, comparator=comparator, ASC=True, track_prov=False,
                                       propagate_prov=False)
        order_handler.set_next_handler.remote(topK_handler)
        # GroupBy operator instantiation
        groupBy_handler_0 = GroupBy.remote(input=None, key=2, value=3, agg_fun=AVG, track_prov=False,
                                           propagate_prov=False)
        groupBy_handler_0.set_next_handler.remote(order_handler)
        groupBy_handler_1 = GroupBy.remote(input=None, key=2, value=3, agg_fun=AVG, track_prov=False,
                                           propagate_prov=False)
        groupBy_handler_1.set_next_handler.remote(order_handler)
        # Join operator instantiation
        join_handler_0 = Join.remote(left_input=None, right_input=None, left_join_attribute=1,
                                     right_join_attribute=0, batch_size=1, track_prov=False,
                                     propagate_prov=False)
        join_handler_0.set_next_handlers.remote([groupBy_handler_0, groupBy_handler_1])
        join_handler_1 = Join.remote(left_input=None, right_input=None, left_join_attribute=1,
                                     right_join_attribute=0, batch_size=1, track_prov=False,
                                     propagate_prov=False)
        join_handler_1.set_next_handlers.remote([groupBy_handler_0, groupBy_handler_1])
        # Select operator instantiation
        select_handler_0 = Select.remote(input=None, predicate=select_predicate_friends, batch_size=1,
                                         track_prov=None, propagate_prov=None)
        select_handler_0.set_next_handlers.remote([join_handler_0, join_handler_1])
        select_handler_1 = Select.remote(input=None, predicate=select_predicate_friends, batch_size=1,
                                         track_prov=None, propagate_prov=None)
        select_handler_1.set_next_handlers.remote([join_handler_0, join_handler_1])
        # Scan friends operator instantiation
        scan_friends_handler_0 = Scan.remote(filepath="../data/Assignment_4_testing_data/friends_01.txt",
                                             filter=None,
                                             batch_size=1,
                                             track_prov=False, propagate_prov=False)
        scan_friends_handler_0.set_next_handlers.remote([select_handler_0, select_handler_1])
        scan_friends_handler_1 = Scan.remote(filepath="../data/Assignment_4_testing_data/friends_02.txt",
                                             filter=None,
                                             batch_size=1,
                                             track_prov=False, propagate_prov=False)
        scan_friends_handler_1.set_next_handlers.remote([select_handler_0, select_handler_1])
        scan_friends_handler_2 = Scan.remote(filepath="../data/Assignment_4_testing_data/friends_03.txt",
                                             filter=None,
                                             batch_size=1,
                                             track_prov=False, propagate_prov=False)
        scan_friends_handler_2.set_next_handlers.remote([select_handler_0, select_handler_1])
        scan_friends_handler_3 = Scan.remote(filepath="../data/Assignment_4_testing_data/friends_04.txt",
                                             filter=None,
                                             batch_size=1,
                                             track_prov=False, propagate_prov=False)
        scan_friends_handler_3.set_next_handlers.remote([select_handler_0, select_handler_1])

        # Scan ratings operator instantiation
        scan_ratings_handler_0 = Scan.remote(filepath="../data/Assignment_4_testing_data/ratings_01.txt",
                                             filter=None, batch_size=1, track_prov=False, propagate_prov=False)
        scan_ratings_handler_0.set_next_handlers.remote([join_handler_0, join_handler_1])
        scan_ratings_handler_1 = Scan.remote(filepath="../data/Assignment_4_testing_data/ratings_02.txt",
                                             filter=None, batch_size=1, track_prov=False, propagate_prov=False)
        scan_ratings_handler_1.set_next_handlers.remote([join_handler_0, join_handler_1])
        scan_ratings_handler_2 = Scan.remote(filepath="../data/Assignment_4_testing_data/ratings_03.txt",
                                             filter=None, batch_size=1, track_prov=False, propagate_prov=False)
        scan_ratings_handler_2.set_next_handlers.remote([join_handler_0, join_handler_1])
        scan_ratings_handler_3 = Scan.remote(filepath="../data/Assignment_4_testing_data/ratings_04.txt",
                                             filter=None, batch_size=1, track_prov=False, propagate_prov=False)
        scan_ratings_handler_3.set_next_handlers.remote([join_handler_0, join_handler_1])

        # Start query execution
        scan_ratings_handler_0.execute.remote()
        scan_ratings_handler_1.execute.remote()
        scan_ratings_handler_2.execute.remote()
        scan_ratings_handler_3.execute.remote()

        scan_friends_handler_0.execute.remote()
        scan_friends_handler_1.execute.remote()
        scan_friends_handler_2.execute.remote()
        scan_friends_handler_3.execute.remote()

        # Blocking call
        time.sleep(5)
        movie_id = ray.get(sink_handler.get_result.remote())
        assert (movie_id[0].tuple == [10])
