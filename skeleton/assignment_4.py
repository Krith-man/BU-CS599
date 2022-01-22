from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import csv
import logging
from typing import List, Tuple
import uuid
import argparse
import os
from pathlib import Path
import ray
import threading
import time
import opentracing
from opentracing.propagation import Format
import logging
from jaeger_client import Config
import sys
from opentracing.ext import tags
from flask import Flask
from flask import request

# Note (john): Make sure you use Python's logger to log
#              information about your program
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Needed arguments
    parser.add_argument("--assignment", help="Assignment number", type=int, required=True)
    parser.add_argument("--task", help="Task number", type=int, required=True)
    parser.add_argument("--friends", help="Path_to_friends_file.txt", type=check_file_existance, required=False)
    parser.add_argument("--ratings", help="Path_to_ratings_file.txt", type=check_file_existance, required=False)
    parser.add_argument("--uid", help="User id.", type=int)
    parser.add_argument("--mid", help="Movie id.", type=int)

    # Parse arguments
    args = parser.parse_args()

    return args


def check_file_existance(x):
    # Checks that file exists but does not open.
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


# Generates unique operator IDs
def _generate_uuid():
    return uuid.uuid4()


# Custom tuple class with optional metadata
class ATuple:
    """Custom tuple.

    Attributes:
        tuple (Tuple): The actual tuple.
        metadata (string): The tuple metadata (e.g. provenance annotations).
        operator (Operator): A handle to the operator that produced the tuple.
    """

    def __init__(self, tuple, metadata, operator):
        self.tuple = tuple
        self.metadata = metadata
        self.operator = operator

    def __repr__(self):
        return str(self.tuple)

    # Returns the lineage of self
    def lineage(self) -> List[ATuple]:
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        return self.operator.lineage([self])

    # Returns the Where-provenance of the attribute at index 'att_index' of self
    def where(self, att_index) -> List[Tuple]:
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        return self.operator.where(att_index, [self])

    # Returns the How-provenance of self
    def how(self) -> str:
        # YOUR CODE HERE (ONLY FOR TASK 3 IN ASSIGNMENT 2)
        if self.metadata is not None:
            return str(self.metadata["How"])
        else:
            return None

    # Returns the input tuples with responsibility \rho >= 0.5 (if any)
    def responsible_inputs(self) -> List[(ATuple, float)]:
        # YOUR CODE HERE (ONLY FOR TASK 4 IN ASSIGNMENT 2)

        tUIDs_list = []
        # List all unique identifiers from all "how" provenance orderBy output tuples
        for i in range(len(self.metadata["OrderByOutput"])):
            for tIDs in self.metadata["OrderByOutput"][i].metadata["TupleIDs"]:
                tUIDs_list.extend(tIDs)

        responsibility_list = []
        # Calculate the responsibility for each unique identifier (tUID)
        for tUID in tUIDs_list:
            ratings = []
            no_pick = []
            average_list = []
            # Check for contingency = 0
            for i in range(len(self.metadata["OrderByOutput"])):
                for index, tUIDs in enumerate(self.metadata["OrderByOutput"][i].metadata["TupleIDs"]):
                    if tUID not in tUIDs:
                        ratings.append(self.metadata["OrderByOutput"][i].metadata["Data"][index])
                    else:
                        # Small optimization:
                        # In case of contingency = 1 we should not pick a
                        # unique identifier that is in same join with tUID we investigate
                        for UID in tUIDs:
                            if UID != tUID:
                                no_pick.append(UID)
                # Get the output average value seperated from the other possible average values
                if i == 0:
                    average = self.AVG(ratings) if ratings else -1
                    ratings = []
                else:
                    average_list.append(self.AVG(ratings) if ratings else -1)
            if average < max(average_list):
                responsibility_list.append(self.UID_info(tUID, "1"))
            # Check for contingency = 1
            else:
                appropriate_UIDs_list = list(tUIDs_list)
                # Remove tUID we investigate as option for contingency as well all
                # tUIDs that are joined with it and belong to no_pick list.
                appropriate_UIDs_list.remove(tUID)
                for UID in no_pick:
                    appropriate_UIDs_list.remove(UID)
                # Start iterate over appropriate UIDs to check if they
                # are appropriate contingency sets for tUID we investigate
                for appropriate_UID in appropriate_UIDs_list:
                    for i in range(len(self.metadata["OrderByOutput"])):
                        ratings = []
                        average_list = []
                        for index, tUIDs in enumerate(self.metadata["OrderByOutput"][i].metadata["TupleIDs"]):
                            if appropriate_UID not in tUIDs:
                                ratings.append(self.metadata["OrderByOutput"][i].metadata["Data"][index])
                        if i == 0:
                            average = self.AVG(ratings) if ratings else -1
                        else:
                            average_list.append(self.AVG(ratings) if ratings else -1)
                    # If average without the contingency set Γ = appropriate_UID is still the greatest
                    # then contingency set Γ = appropriate_UID has passed the first step of actual case q(D-Γ) |= r
                    if average >= max(average_list):
                        for i in range(len(self.metadata["OrderByOutput"])):
                            ratings = []
                            average_list = []
                            for index, tUIDs in enumerate(self.metadata["OrderByOutput"][i].metadata["TupleIDs"]):
                                if appropriate_UID not in tUIDs:
                                    if tUID not in tUIDs:
                                        ratings.append(self.metadata["OrderByOutput"][i].metadata["Data"][index])
                            if i == 0:
                                average = self.AVG(ratings) if ratings else -1
                            else:
                                average_list.append(self.AVG(ratings) if ratings else -1)
                        # If average without the contingency set Γ = UID union tUID we investigate
                        # is not the greatest then contingency set Γ = UID has passed the second step of
                        # actual case q(D-Γ U {tUID}) !|= r
                        if average < max(average_list):
                            responsibility_list.append(self.UID_info(tUID, "0.5"))
                            break
        return responsibility_list

    # Returns the average value for calculating the responsibility in TASK IV
    def AVG(self, ratings):
        sum = 0
        for rating in ratings:
            sum += int(rating)
        return sum / len(ratings)

    # Returns the infos for each tuples through its position to the file in TASK IV
    def UID_info(self, tUID, responsibility_value):
        if tUID[0] == 'R':
            f = open(args.ratings)
        else:
            f = open(args.friends)
        lines = f.read().splitlines()
        return [ATuple([lines[int(tUID[1])]], metadata=None, operator=None), responsibility_value]


# Data operator
class Operator:
    """Data operator (parent class).

    Attributes:
        id (string): Unique operator ID.
        name (string): Operator name.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    def __init__(self, id=None, name=None, track_prov=False,
                 propagate_prov=False):
        self.id = _generate_uuid() if id is None else id
        self.name = "Undefined" if name is None else name
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.jaeger_name = "Assignment_04_local"
        self.tracer = None
        logger.debug("Created {} operator with id {}".format(self.name,
                                                             self.id))

    # NOTE (john): Must be implemented by the subclasses
    def get_next(self):
        logger.error("Method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def lineage(self, tuples: List[ATuple]) -> List[List[ATuple]]:
        logger.error("Lineage method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def where(self, att_index: int, tuples: List[ATuple]) -> List[List[ATuple]]:
        logger.error("Where-provenance method not implemented!")

    def init_tracer(self, service):
        logging.getLogger('').handlers = []
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)

        config = Config(
            config={
                'sampler': {
                    'type': 'const',
                    'param': 1,
                },
                'logging': True,
            },
            service_name=service,
        )

        return config.new_tracer()

# Sink operator
@ray.remote
class Sink(Operator):
    """Sink operator.
    Attributes:
    """

    # Initializes Sink operator
    def __init__(self):
        Operator.__init__(self, name="Sink")
        # YOUR CODE HERE
        self.result = []

    def execute(self, input_tuples, span_ctx):
        tracer = self.init_tracer(self.jaeger_name)
        with tracer.start_span(operation_name=self.name, references=opentracing.child_of(span_ctx)) as span:
            span.set_tag(key="pid",value= os.getpid())
            if input_tuples is not None:
                self.result.extend(input_tuples)
            return True

    def get_result(self):
        return self.result


# Scan operator
@ray.remote
class Scan(Operator):
    """Scan operator.

    Attributes:
        filepath (string): The path to the input file.
        filter (function): An optional user-defined filter.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    # Initializes scan operator
    def __init__(self, filepath, filter, batch_size, track_prov,
                 propagate_prov):
        Operator.__init__(self, name="Scan", track_prov=track_prov,
                          propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.filepath = filepath
        self.filter = filter
        self.batch_size = batch_size
        self.next_position = 0
        self.execute_position = 0
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.inout_attr_mappings = {}
        self.handlers = []

    # Returns next batch of tuples in given file (or None if file exhausted)
    def get_next(self):
        # YOUR CODE HERE
        counter_batch = 0
        output_tuples = []

        with open(self.filepath) as f:
            next(f)
            for i, t in enumerate(csv.reader(f, delimiter=' ')):
                # Start reading from where you stopped
                if i == self.next_position:
                    dict_metadata = {}
                    if self.propagate_prov:
                        if "Friends" in self.filepath:
                            dict_metadata.update({"How": Path(self.filepath).stem[0] + str(i + 1)})
                            dict_metadata.update({"TupleIDs": [Path(self.filepath).stem[0] + str(i + 1)]})
                            dict_metadata.update({"Data": -1})
                        elif "Ratings" in self.filepath:
                            dict_metadata.update({"How": Path(self.filepath).stem[0] + str(i + 1) + "@" + str(t[2])})
                            dict_metadata.update({"TupleIDs": [Path(self.filepath).stem[0] + str(i + 1)]})
                            dict_metadata.update({"Data": t[2]})
                        else:
                            dict_metadata.update({"How": Path(self.filepath).stem[0] + str(i + 1)})
                            # Useful only for TaskIV
                            dict_metadata.update({"TupleIDs": []})
                            dict_metadata.update({"Data": -1})
                    input_tuple = ATuple(t, metadata=dict_metadata if self.propagate_prov else None, operator=self)
                    if self.filter is None or self.filter(input_tuple):
                        output_tuples.append(input_tuple)
                        if self.track_prov:
                            # Dictionary inout_attr_mappings
                            # Key -> output_tuple of Scan
                            # Value -> Position of tuple in file
                            self.inout_attr_mappings.update({input_tuple: i})
                        counter_batch += 1
                    self.next_position += 1
                    # Read the batch of tuples
                    if counter_batch == self.batch_size:
                        break
        if not output_tuples:
            return None
        return output_tuples

    def execute(self):
        tracer = self.init_tracer(self.jaeger_name)
        with tracer.start_span(operation_name=self.name) as span:
            span.set_tag(key="pid", value=os.getpid())

            counter_batch_0 = 0
            counter_batch_1 = 0
            bucket_handler_0 = []
            bucket_handler_1 = []

            with open(self.filepath) as f:
                next(f)
                for i, t in enumerate(csv.reader(f, delimiter=' ')):
                    input_tuple = ATuple(t, metadata=None, operator=self)
                    if self.filter is None or self.filter(input_tuple):
                        if "friends" in Path(self.filepath).stem:
                            # Apply hash function (mod 2) at Friends.UID2 attribute to find in
                            # which bucket handler the tuple will be stored
                            if int(t[1]) % 2 == 0:
                                bucket_handler_0.append(input_tuple)
                                counter_batch_0 += 1
                            else:
                                bucket_handler_1.append(input_tuple)
                                counter_batch_1 += 1
                            if counter_batch_0 == self.batch_size:
                                self.handlers[0].execute.remote(bucket_handler_0, 0, span.context)
                                bucket_handler_0 = []
                                counter_batch_0 = 0
                            if counter_batch_1 == self.batch_size:
                                self.handlers[1].execute.remote(bucket_handler_1, 1, span.context)
                                bucket_handler_1 = []
                                counter_batch_1 = 0
                        if "ratings" in Path(self.filepath).stem:
                            t[1] = int(t[1])
                            # Apply hash function (mod 2) at Ratings.UID attribute to find in
                            # which bucket handler the tuple will be stored
                            if int(t[0]) % 2 == 0:
                                bucket_handler_0.append(input_tuple)
                                counter_batch_0 += 1
                            else:
                                bucket_handler_1.append(input_tuple)
                                counter_batch_1 += 1
                            # File ID = 1 for Ratings table (Indicates to "Join" operator
                            # that input data come from Ratings table)
                            if counter_batch_0 == self.batch_size:
                                self.handlers[0].execute.remote(bucket_handler_0, 1, span.context)
                                bucket_handler_0 = []
                                counter_batch_0 = 0
                            if counter_batch_1 == self.batch_size:
                                self.handlers[1].execute.remote(bucket_handler_1, 1, span.context)
                                bucket_handler_1 = []
                                counter_batch_1 = 0

            if "ratings" in Path(self.filepath).stem:
                # Check if bucket_handler_0 has still tuples to push to output and if yes push them.
                if bucket_handler_0:
                    self.handlers[0].execute.remote(bucket_handler_0, 1, span.context)
                # Check if bucket_handler_1 has still tuples to push to output and if yes push them.
                if bucket_handler_1:
                    self.handlers[1].execute.remote(bucket_handler_1, 1, span.context)
                # After pushing last tuples push None as indicator that input has been exhausted.
                self.handlers[0].execute.remote(None, 1, span.context)
                self.handlers[1].execute.remote(None, 1, span.context)
            if "friends" in Path(self.filepath).stem:
                # Check if bucket_handler_0 has still tuples to push to output and if yes push them.
                if bucket_handler_0:
                    self.handlers[0].execute.remote(bucket_handler_0, 0, span.context)
                # Check if bucket_handler_1 has still tuples to push to output and if yes push them.
                if bucket_handler_1:
                    self.handlers[1].execute.remote(bucket_handler_1, 1, span.context)
                # After pushing last tuples push None as indicator that input has been exhausted.
                self.handlers[0].execute.remote(None, 0, span.context)
                self.handlers[1].execute.remote(None, 1, span.context)

        return True

    def set_next_handlers(self, handlers):
        self.handlers = handlers

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        output_lineage = []
        for t in tuples:
            output_lineage.extend([t])
        return output_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        output_where_lineage = []
        for t in tuples:
            output_where_lineage.extend(
                [ATuple([Path(self.filepath).stem, self.inout_attr_mappings[t], t.tuple, t.tuple[att_index]],
                        metadata=None,
                        operator=self)])
        return output_where_lineage


# Equi-join operator
@ray.remote
class Join(Operator):
    """Equi-join operator.

    Attributes:
        left_input (Operator): A handle to the left input.
        right_input (Operator): A handle to the left input.
        left_join_attribute (int): The index of the left join attribute.
        right_join_attribute (int): The index of the right join attribute.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    # Initializes join operator
    def __init__(self, left_input, right_input, left_join_attribute,
                 right_join_attribute, batch_size,
                 track_prov,
                 propagate_prov=False):
        Operator.__init__(self, name="Join", track_prov=track_prov,
                          propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.left_input = left_input
        self.batch_size = batch_size
        self.right_input = right_input
        self.left_join_attribute = left_join_attribute
        self.right_join_attribute = right_join_attribute
        self.left_hash_table = {}
        self.left_input_flag = 0
        self.join_buffer = []
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.inout_mappings = {}
        self.inout_left_attr_mappings = {}
        self.inout_right_attr_mappings = {}
        self.execute_left_hash_table = {}
        self.execute_right_hash_table = {}
        self.buffer_handler_0 = []
        self.buffer_handler_1 = []
        self.num_finished = 0

    # Returns next batch of joined tuples (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        counter_batch = 0
        output_tuples = []
        counter_attr = 0

        # Initially build the left hash table on the left join attribute for all the left tuples till input is exhausted
        if self.left_input_flag == 0:
            input_left_tuples = ray.get(self.left_input.get_next.remote())
            while input_left_tuples is not None:
                for t in input_left_tuples:
                    if t.tuple[self.left_join_attribute] in self.left_hash_table:
                        t.operator = self.left_input
                        self.left_hash_table[t.tuple[self.left_join_attribute]].append(t)
                    else:
                        t.operator = self.left_input
                        self.left_hash_table.update(
                            {t.tuple[self.left_join_attribute]: [t]})
                input_left_tuples = ray.get(self.left_input.get_next.remote())
            # After building the left hash table revert the flag to bypass a possible new building.
            self.left_input_flag = 1

        # Check if join buffer is empty and if not output its content
        if self.join_buffer:
            for i, t in enumerate(self.join_buffer):
                # Add join tuple to output
                output_tuples.append(t)
                counter_batch += 1
                if counter_batch == self.batch_size:
                    # Remove as many join tuples added to output_tuples
                    self.join_buffer = self.join_buffer[counter_batch:]
                    return output_tuples

            # Remove as many join tuples added to output_tuples
            self.join_buffer = self.join_buffer[counter_batch:]

        input_right_tuples = ray.get(self.right_input.get_next.remote())
        # Check if input has been exhausted
        if input_right_tuples is None and counter_batch > 0:
            return output_tuples
        elif input_right_tuples is None and counter_batch == 0:
            return None
        else:
            for right_t in input_right_tuples:
                # Equi-join on the same attribute
                if right_t.tuple[self.right_join_attribute] in self.left_hash_table:
                    # Build the join tuple for all left tuples containing the common attribute
                    for left_t in self.left_hash_table[right_t.tuple[self.right_join_attribute]]:
                        counter_batch += 1
                        dict_metadata = {}
                        if self.propagate_prov:
                            dict_metadata.update(
                                {"TupleIDs": left_t.metadata["TupleIDs"] + right_t.metadata["TupleIDs"]})
                            dict_metadata.update(
                                {"Data": right_t.metadata["Data"]})
                            dict_metadata.update({"How": left_t.metadata["How"] + "*" + right_t.metadata["How"]})
                        # Concatenate left and right tuples without the right join attribute
                        generated_tuple = ATuple(left_t.tuple +
                                                 right_t.tuple[:self.right_join_attribute] +
                                                 right_t.tuple[self.right_join_attribute + 1:],
                                                 metadata=dict_metadata if self.propagate_prov else None, operator=self)
                        if self.track_prov:
                            self.inout_mappings.update({generated_tuple: [right_t, left_t]})
                            # Build the inout_left_attr_mappings and inout_right_attr_mappings
                            # to map the indexes of attributes at generated tuple to right and left input
                            for i in range(len(generated_tuple.tuple)):
                                if i < len(left_t.tuple):
                                    self.inout_left_attr_mappings.update({i: i})
                                    # Common attribute in left and right input tables
                                    if i == self.left_join_attribute:
                                        self.inout_right_attr_mappings.update({i: self.right_join_attribute})
                                else:
                                    if counter_attr == self.right_join_attribute:
                                        # Bypass the common attribute in right input table
                                        counter_attr += 1
                                    self.inout_right_attr_mappings.update({i: counter_attr})
                                    counter_attr += 1
                            counter_attr = 0
                        # If exceed the self.batch_size save the rest join tuples to join buffer
                        if counter_batch > self.batch_size:
                            self.join_buffer.append(generated_tuple)
                        else:
                            output_tuples.append(generated_tuple)

            return output_tuples

    def execute(self, input_tuples, file_id, span_ctx):
        """
        input_tuples (List[]): A list of Atuples coming from the Scan operator.
        file_id (int): Identifier depicting if the input_tuples come from Friends table (id =0) or Ratings table (id =1)
        span_ctx(SpanContext Object): Span context from the parent span of "Join" operator
        """
        tracer = self.init_tracer(self.jaeger_name)
        with tracer.start_span(operation_name=self.name, references=opentracing.child_of(span_ctx)) as span:
            span.set_tag(key="pid", value=os.getpid())

            bucket_handler_0 = []
            bucket_handler_1 = []
            counter_batch_0 = 0
            counter_batch_1 = 0

            # Check if buffer_handler_0 is empty and if not output its content
            if self.buffer_handler_0:
                for i, t in enumerate(self.buffer_handler_0):
                    # Add projection tuple to output
                    bucket_handler_0.append(t)
                    counter_batch_0 += 1
                    if counter_batch_0 == self.batch_size:
                        self.handlers[0].execute.remote(bucket_handler_0, span.context)
                        # Remove as many selection tuples added to bucket_handler_0
                        self.buffer_handler_0 = self.buffer_handler_0[counter_batch_0:]

                # Remove as many selection tuples added to bucket_handler
                self.buffer_handler_0 = self.buffer_handler_0[counter_batch_0:]

            # Check if buffer_handler_1 is empty and if not output its content
            if self.buffer_handler_1:
                for i, t in enumerate(self.buffer_handler_1):
                    # Add projection tuple to output
                    bucket_handler_1.append(t)
                    counter_batch_1 += 1
                    if counter_batch_1 == self.batch_size:
                        self.handlers[1].execute.remote(bucket_handler_1, span.context)
                        # Remove as many selection tuples added to bucket_handler_1
                        self.buffer_handler_1 = self.buffer_handler_1[counter_batch_1:]

                # Remove as many selection tuples added to bucket_handler
                self.buffer_handler_1 = self.buffer_handler_1[counter_batch_1:]

            # Check if input has been exhausted
            if input_tuples is None:
                self.num_finished += 1
                # Send last tuples stored at bucket_handler_0
                if counter_batch_0 > 0:
                    self.handlers[0].execute.remote(bucket_handler_0, span.context)
                # Send last tuples stored at bucket_handler_1
                if counter_batch_1 > 0:
                    self.handlers[1].execute.remote(bucket_handler_1, span.context)
                # Check if 4 instances of Scan Rating table and 2 Select Friends table has been finished and
                # send None to indicate that input has been exhausted
                if self.num_finished == 6:
                    self.handlers[0].execute.remote(None, span.context)
                    self.handlers[1].execute.remote(None, span.context)
            else:
                counter_batch_0 = 0
                counter_batch_1 = 0
                bucket_handler_0 = []
                bucket_handler_1 = []
                # Check if input tuples are coming from Friends table
                if file_id == 0:
                    # Store the data to left_hash_table (Friends)
                    for t in input_tuples:
                        if t.tuple[self.left_join_attribute] in self.execute_left_hash_table:
                            self.execute_left_hash_table[t.tuple[self.left_join_attribute]].append(t)
                        else:
                            self.execute_left_hash_table.update({t.tuple[self.left_join_attribute]: [t]})
                    # Check if dictionary of Ratings is not empty and apply "Join" logic
                    if self.execute_right_hash_table:
                        for left_t in input_tuples:
                            # X-join on the same attribute
                            if left_t.tuple[self.left_join_attribute] in self.execute_right_hash_table:
                                # Build the join tuple for all right tuples containing the common attribute
                                for right_t in self.execute_right_hash_table[left_t.tuple[self.left_join_attribute]]:
                                    # Concatenate left and right tuples without the right join attribute
                                    generated_tuple = ATuple(left_t.tuple +
                                                             right_t.tuple[:self.right_join_attribute] +
                                                             right_t.tuple[self.right_join_attribute + 1:],
                                                             metadata=None,
                                                             operator=self)
                                    # Apply hash function (mod 2) at generated_tuple.MID attribute to find in
                                    # which bucket handler the tuple will be stored
                                    if int(generated_tuple.tuple[2]) % 2 == 0:
                                        # If exceed the self.batch_size save the rest join tuples to bucket_0 buffer
                                        if counter_batch_0 > self.batch_size:
                                            self.buffer_handler_0.append(generated_tuple)
                                        else:
                                            bucket_handler_0.append(generated_tuple)
                                        counter_batch_0 += 1
                                    else:
                                        # If exceed the self.batch_size save the rest join tuples to bucket_1 buffer
                                        if counter_batch_1 > self.batch_size:
                                            self.buffer_handler_1.append(generated_tuple)
                                        else:
                                            bucket_handler_1.append(generated_tuple)
                                        counter_batch_1 += 1
                        # Call "GroupBy" handlers
                        if bucket_handler_0:
                            self.handlers[0].execute.remote(bucket_handler_0, span.context)
                        if bucket_handler_1:
                            self.handlers[1].execute.remote(bucket_handler_1, span.context)
                # Check if input tuples are coming from Ratings table
                else:
                    # Store the data to right_hash_table (Ratings)
                    for t in input_tuples:
                        if t.tuple[self.right_join_attribute] in self.execute_right_hash_table:
                            self.execute_right_hash_table[t.tuple[self.right_join_attribute]].append(t)
                        else:
                            self.execute_right_hash_table.update({t.tuple[self.right_join_attribute]: [t]})
                    # Check if dictionary of Friends is not empty and apply "Join" logic
                    if self.execute_left_hash_table:
                        for right_t in input_tuples:
                            # X-join on the same attribute
                            if right_t.tuple[self.right_join_attribute] in self.execute_left_hash_table:
                                # Build the join tuple for all left tuples containing the common attribute
                                for left_t in self.execute_left_hash_table[right_t.tuple[self.right_join_attribute]]:
                                    # Concatenate left and right tuples without the right join attribute
                                    generated_tuple = ATuple(left_t.tuple +
                                                             right_t.tuple[:self.right_join_attribute] +
                                                             right_t.tuple[self.right_join_attribute + 1:],
                                                             metadata=None,
                                                             operator=self)
                                    # Apply hash function (mod 2) at generated_tuple.MID attribute to find in
                                    # which bucket handler the tuple will be stored
                                    if int(generated_tuple.tuple[2]) % 2 == 0:
                                        # If exceed the self.batch_size save the rest join tuples to bucket_0 buffer
                                        if counter_batch_0 > self.batch_size:
                                            self.buffer_handler_0.append(generated_tuple)
                                        else:
                                            bucket_handler_0.append(generated_tuple)
                                        counter_batch_0 += 1
                                    else:
                                        # If exceed the self.batch_size save the rest join tuples to bucket_1 buffer
                                        if counter_batch_1 > self.batch_size:
                                            self.buffer_handler_1.append(generated_tuple)
                                        else:
                                            bucket_handler_1.append(generated_tuple)
                                        counter_batch_1 += 1
                        # Call "GroupBy" handlers
                        if bucket_handler_0:
                            self.handlers[0].execute.remote(bucket_handler_0, span.context)
                        if bucket_handler_1:
                            self.handlers[1].execute.remote(bucket_handler_1, span.context)
        return True

    def set_next_handlers(self, handlers):
        self.handlers = handlers

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        input_right_tuples = []
        input_left_tuples = []
        output_lineage = []
        for t in tuples:
            # Take all right traversal tuples
            right_operator = self.inout_mappings[t][0].operator
            input_right_tuples.append(self.inout_mappings[t][0])
            # Take all left traversal tuples
            left_operator = self.inout_mappings[t][1].operator
            input_left_tuples.append(self.inout_mappings[t][1])
        # Build the output result based on right and left traversals
        output_lineage.extend(right_operator.lineage(input_right_tuples))
        output_lineage.extend(left_operator.lineage(input_left_tuples))
        return output_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        input_right_tuples = []
        input_left_tuples = []
        output_where_lineage = []
        for t in tuples:
            # If att_index is the join attribute we check both right and left trees
            if att_index in self.inout_right_attr_mappings and att_index in self.inout_left_attr_mappings:
                # Right traversal
                right_operator = self.inout_mappings[t][0].operator
                right_att_index = self.inout_right_attr_mappings[att_index]
                input_right_tuples.append(self.inout_mappings[t][0])
                # Left traversal
                left_operator = self.inout_mappings[t][1].operator
                left_att_index = self.inout_left_attr_mappings[att_index]
                input_left_tuples.append(self.inout_mappings[t][1])
            else:
                if att_index in self.inout_right_attr_mappings:
                    right_operator = self.inout_mappings[t][0].operator
                    right_att_index = self.inout_right_attr_mappings[att_index]
                    input_right_tuples.append(self.inout_mappings[t][0])
                else:
                    left_operator = self.inout_mappings[t][1].operator
                    left_att_index = self.inout_left_attr_mappings[att_index]
                    input_left_tuples.append(self.inout_mappings[t][1])

        if att_index in self.inout_right_attr_mappings and att_index in self.inout_left_attr_mappings:
            output_where_lineage.extend(right_operator.where(right_att_index, input_right_tuples))
            output_where_lineage.extend(left_operator.where(left_att_index, input_left_tuples))
        else:
            if att_index in self.inout_right_attr_mappings:
                output_where_lineage.extend(right_operator.where(right_att_index, input_right_tuples))
            else:
                output_where_lineage.extend(left_operator.where(left_att_index, input_left_tuples))
        return output_where_lineage


# Project operator
@ray.remote
class Project(Operator):
    """Project operator.

    Attributes:
        input (Operator): A handle to the input.
        fields_to_keep (List(int)): A list of attribute indices to keep.
        If empty, the project operator behaves like an identity map, i.e., it
        produces and output that is identical to its input.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    # Initializes project operator
    def __init__(self, input, fields_to_keep, batch_size, track_prov,
                 propagate_prov):
        Operator.__init__(self, name="Project", track_prov=track_prov,
                          propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        self.fields_to_keep = fields_to_keep
        self.batch_size = batch_size
        self.projection_buffer = []
        self.execute_projection_buffer = []
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.inout_mappings = {}
        # Dictionary inout_attr_mappings
        # Key -> attribute index of generated tuple
        # Value ->  attribute index of input tuple
        self.inout_attr_mappings = {}

    # Return next batch of projected tuples (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        list_attributes = []
        output_tuples = []
        counter_batch = 0
        counter_attr = 0

        # Check if projection_buffer is empty and if not output its content
        if self.projection_buffer:
            for i, t in enumerate(self.projection_buffer):
                # Add projection tuple to output
                output_tuples.append(t)
                counter_batch += 1
                if counter_batch == self.batch_size:
                    # Remove as many projection tuples added to output_tuples
                    self.projection_buffer = self.projection_buffer[counter_batch:]
                    return output_tuples

            # Remove as many projection tuples added to output_tuples
            self.projection_buffer = self.projection_buffer[counter_batch:]

        input_tuples = ray.get(self.input.get_next.remote())
        # Check if input has been exhausted
        if input_tuples is None and counter_batch > 0:
            return output_tuples
        elif input_tuples is None and counter_batch == 0:
            return None
        else:
            for t in input_tuples:
                # Project whole tuple
                if not self.fields_to_keep:
                    generated_tuple = ATuple(t.tuple, metadata=t.metadata if self.propagate_prov else None,
                                             operator=self)
                    if self.track_prov:
                        self.inout_mappings.update({generated_tuple: t})
                        for i in range(len(t.tuple)):
                            self.inout_attr_mappings.update({i: i})
                # Build the projected tuple with "fields_to_keep" attributes
                else:
                    for attribute in self.fields_to_keep:
                        list_attributes.append(t.tuple[attribute])
                        if self.track_prov:
                            self.inout_attr_mappings.update({counter_attr: attribute})
                        counter_attr += 1
                    counter_attr = 0
                    generated_tuple = ATuple(list_attributes, metadata=t.metadata if self.propagate_prov else None,
                                             operator=self)
                    if self.track_prov:
                        self.inout_mappings.update({generated_tuple: t})
                    list_attributes = []
                # Write input tuples to projection_buffer to read them in next call of get_next()
                if counter_batch > self.batch_size:
                    self.projection_buffer.append(generated_tuple)
                # Write input tuples to output_tuples
                else:
                    output_tuples.append(generated_tuple)
                counter_batch += 1

            return output_tuples

    def execute(self, input_tuples, span_ctx):
        """
        input_tuples (List[]): A list of Atuples coming from the TopK operator.
        span_ctx(SpanContext Object): Span context from the parent span of "Project" operator
        """
        tracer = self.init_tracer(self.jaeger_name)
        with tracer.start_span(operation_name=self.name, references=opentracing.child_of(span_ctx)) as span:
            span.set_tag(key="pid", value=os.getpid())
            bucket_handler = []
            counter_batch = 0
            list_attributes = []

            # Check if projection_buffer is empty and if not output its content
            if self.execute_projection_buffer:
                for i, t in enumerate(self.execute_projection_buffer):
                    bucket_handler.append(t)
                    counter_batch += 1
                    if counter_batch == self.batch_size:
                        # Remove as many projection tuples added to bucket_handler
                        self.execute_projection_buffer = self.execute_projection_buffer[counter_batch:]
                        # Call "Sink" handler
                        self.handler.execute.remote(bucket_handler, span.context)

                # Remove as many projection tuples added to bucket_handler
                self.execute_projection_buffer = self.execute_projection_buffer[counter_batch:]

            # Check if input has been exhausted
            if input_tuples is None:
                if counter_batch > 0:
                    # Call "Sink" handler
                    self.handler.execute.remote(bucket_handler, span.context)
                    self.handler.execute.remote(None, span.context)
                else:
                    # Call "Sink" handler
                    self.handler.execute.remote(None, span.context)
            else:
                for t in input_tuples:
                    # Project whole tuple
                    if not self.fields_to_keep:
                        generated_tuple = ATuple(t.tuple, metadata=None, operator=self)
                    # Build the projected tuple with "fields_to_keep" attributes
                    else:
                        for attribute in self.fields_to_keep:
                            list_attributes.append(t.tuple[attribute])
                        generated_tuple = ATuple(list_attributes, metadata=None, operator=self)
                        list_attributes = []
                    # Write input tuples to projection_buffer to read them in next call of execute()
                    if counter_batch > self.batch_size:
                        self.execute_projection_buffer.append(generated_tuple, span.context)
                    # Write input tuples to output_tuples
                    else:
                        bucket_handler.append(generated_tuple)
                    counter_batch += 1
                # Call "Sink" handler
                self.handler.execute.remote(bucket_handler, span.context)
            return True

    def set_next_handler(self, handler):
        self.handler = handler

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        input_tuples = []
        output_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t].operator
            input_tuples.append(self.inout_mappings[t])
        output_lineage.extend(next_operator.lineage(input_tuples))
        return output_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        input_tuples = []
        output_where_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t].operator
            input_tuples.append(self.inout_mappings[t])
        attribute = self.inout_attr_mappings[att_index]
        output_where_lineage.extend(next_operator.where(attribute, input_tuples))
        return output_where_lineage


# Group-by operator
@ray.remote
class GroupBy(Operator):
    """Group-by operator.

    Attributes:
        input (Operator): A handle to the input
        key (int): The index of the key to group tuples.
        value (int): The index of the attribute we want to aggregate.
        agg_fun (function): The aggregation function (e.g. AVG)
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    # Initializes average operator
    def __init__(self, input, key, value, agg_fun, track_prov,
                 propagate_prov):
        Operator.__init__(self, name="GroupBy", track_prov=track_prov,
                          propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        self.key = key
        self.value = value
        self.agg_fun = agg_fun
        self.group_buffer = []
        self.execute_group_buffer = []
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.inout_mappings = {}
        self.num_finished_Joins = 0

    # Returns aggregated value per distinct key in the input (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        dict_group = {}
        group_list = []
        how_list = []
        # Check if we have assigned value to group_buffer to guarantee that it is done
        if self.group_buffer:
            return None

        input_tuples = ray.get(self.input.get_next.remote())
        # While input has not been exhausted keep adding to group_buffer
        while input_tuples is not None:
            for t in input_tuples:
                self.group_buffer.append(t)
            input_tuples = ray.get(self.input.get_next.remote())

        # Simple average on self.value attribute
        if self.key == -1:
            dict_metadata = {}
            how_list = []
            for tuple in self.group_buffer:
                how_list.append(tuple.metadata["How"])
            # Make a list of all tuple's metadata that contribute to the final aggregation result
            dict_metadata.update({"How": self.agg_fun.__name__ + "(" + str(how_list) + ")"})
            generated_tuple = ATuple(tuple=[self.agg_fun(self.group_buffer, self.value)],
                                     metadata=dict_metadata if self.propagate_prov else None, operator=self)
            if self.track_prov:
                self.inout_mappings.update({generated_tuple: self.group_buffer})
            group_list.append(generated_tuple)
        # Group by average on self.value attribute and self.key distinct keys
        else:
            # Build a dictionary with keys being the different values of self.key and values the
            # list of respective tuples
            for t in self.group_buffer:
                if t.tuple[self.key] in dict_group:
                    t.operator = self.input
                    dict_group[t.tuple[self.key]].append(t)
                else:
                    t.operator = self.input
                    dict_group.update({t.tuple[self.key]: [t]})

            # Lookup each key of dictionary and calculate the average value for each key through self.agg_fun
            for key, list_tuples in dict_group.items():
                dict_metadata = {}
                how_list = []
                if self.propagate_prov:
                    dict_metadata.update({"TupleIDs": []})
                    dict_metadata.update({"Data": []})
                    for tuple in list_tuples:
                        how_list.append(tuple.metadata["How"])
                        dict_metadata["TupleIDs"].append(tuple.metadata["TupleIDs"])
                        dict_metadata["Data"].append(tuple.metadata["Data"])
                    # Make a list of all tuple's metadata that contribute to the final aggregation result
                    dict_metadata.update({"How": self.agg_fun.__name__ + "(" + str(how_list) + ")"})
                generated_tuple = ATuple(tuple=[key, self.agg_fun(list_tuples, self.value)],
                                         metadata=dict_metadata if self.propagate_prov else None,
                                         operator=self)
                if self.track_prov:
                    self.inout_mappings.update({generated_tuple: list_tuples})
                group_list.append(generated_tuple)

        return group_list

    def execute(self, input_tuples, span_ctx):
        """
        input_tuples (List[]): A list of Atuples coming from the Join operator.
        span_ctx(SpanContext Object): Span context from the parent span of "GroupBy" operator
        """
        tracer = self.init_tracer(self.jaeger_name)
        with tracer.start_span(operation_name=self.name, references=opentracing.child_of(span_ctx)) as span:
            span.set_tag(key="pid", value=os.getpid())

            bucket_handler = []
            dict_group = {}

            if input_tuples is not None:
                self.execute_group_buffer.extend(input_tuples)
            else:
                self.num_finished_Joins += 1
                if self.num_finished_Joins == 2:
                    # Simple average on self.value attribute
                    if self.key == -1:
                        generated_tuple = ATuple(tuple=[self.agg_fun(self.group_buffer, self.value)],
                                                 metadata=None, operator=self)
                        bucket_handler.append(generated_tuple)
                    # Group by average on self.value attribute and self.key distinct keys
                    else:
                        # Build a dictionary with keys being the different values of self.key and values the
                        # list of respective tuples
                        for t in self.execute_group_buffer:
                            if t.tuple[self.key] in dict_group:
                                dict_group[t.tuple[self.key]].append(t)
                            else:
                                dict_group.update({t.tuple[self.key]: [t]})

                        # Lookup each key of dictionary and calculate the
                        # average value for each key through self.agg_fun
                        for key, list_tuples in dict_group.items():
                            generated_tuple = ATuple(tuple=[key, self.agg_fun(list_tuples, self.value)], metadata=None,
                                                     operator=self)
                            bucket_handler.append(generated_tuple)
                        # Call "OrderBy" handler
                        self.handler.execute.remote(bucket_handler, span.context)
                        self.handler.execute.remote(None, span.context)
            return True

    def set_next_handler(self, handler):
        self.handler = handler

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        output_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t][0].operator
            input_tuples = self.inout_mappings[t]
            output_lineage.extend(next_operator.lineage(input_tuples))
        return output_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        output_where_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t][0].operator
            input_tuples = self.inout_mappings[t]
            output_where_lineage.extend(next_operator.where(self.value, input_tuples))
        return output_where_lineage


# Custom histogram operator
@ray.remote
class Histogram(Operator):
    """Histogram operator.

    Attributes:
        input (Operator): A handle to the input
        key (int): The index of the key to group tuples. The operator outputs
        the total number of tuples per distinct key.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    # Initializes histogram operator
    def __init__(self, input, key, track_prov, propagate_prov):
        Operator.__init__(self, name="Histogram",
                          track_prov=track_prov,
                          propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        self.key = key
        self.histogram_buffer = []
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.inout_mappings = {}

    # Returns histogram (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        dict_rating = {}
        dict_prov = {}
        histogram_list = []

        # Check if we have assigned value to self.histogram_buffer to guarantee that it is done
        if self.histogram_buffer:
            return None

        input_tuples = ray.get(self.input.get_next.remote())
        # While input has not been exhausted keep adding to histogram_buffer
        while input_tuples is not None:
            for t in input_tuples:
                t.operator = self.input
                self.histogram_buffer.append(t)
            input_tuples = ray.get(self.input.get_next.remote())

        for t in self.histogram_buffer:
            t.operator = self.input
            # dict_rating -> dict for counting how many tuples exists per "t.tuple[self.key]" key
            # dict_prov -> dict for tracking which input tuples are responsible per "t.tuple[self.key]" key
            if t.tuple[self.key] in dict_rating:
                dict_rating[t.tuple[self.key]] += 1
                dict_prov[t.tuple[self.key]].append(t)
            else:
                dict_rating.update({t.tuple[self.key]: 1})
                dict_prov.update({t.tuple[self.key]: [t]})

        for key, value in dict_rating.items():
            dict_metadata = {}
            how_list = []
            for tuple in dict_prov[key]:
                how_list.append(tuple.metadata["How"])
            dict_metadata.update({"How": self.name + "(" + str(how_list) + ")"})
            generated_tuple = ATuple([key, value], metadata=dict_metadata if self.propagate_prov else None,
                                     operator=self)
            if self.track_prov:
                self.inout_mappings.update({generated_tuple: dict_prov[key]})
            histogram_list.append(generated_tuple)
        return histogram_list

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        input_tuples = []
        output_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t][0].operator
            input_tuples = self.inout_mappings[t]
            output_lineage.extend(next_operator.lineage(input_tuples))
        return output_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        output_where_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t][0].operator
            input_tuples = self.inout_mappings[t]
            output_where_lineage.extend(next_operator.where(self.key, input_tuples))
        return output_where_lineage


# Order by operator
@ray.remote
class OrderBy(Operator):
    """OrderBy operator.

    Attributes:
        input (Operator): A handle to the input
        comparator (function): The user-defined comparator used for sorting the
        input tuples.
        ASC (bool): True if sorting in ascending order, False otherwise.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    # Initializes order-by operator
    def __init__(self, input, comparator, ASC, track_prov,
                 propagate_prov):
        Operator.__init__(self, name="OrderBy",
                          track_prov=track_prov,
                          propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        self.comparator = comparator
        self.ASC = ASC
        self.order_buffer = []
        self.execute_order_buffer = []
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.inout_mappings = {}
        self.num_finished_GroupBy = 0

    # Returns the sorted input (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        # Check if we have assigned value to self.order_buffer to guarantee that it is done
        if self.order_buffer:
            return None

        input_tuples = ray.get(self.input.get_next.remote())
        # While input has not been exhausted keep adding to order_buffer
        while input_tuples is not None:
            for t in input_tuples:
                generated_tuple = ATuple(t.tuple, metadata=t.metadata if self.propagate_prov else None, operator=self)
                if self.track_prov:
                    self.inout_mappings.update({generated_tuple: t})
                self.order_buffer.append(generated_tuple)
            input_tuples = ray.get(self.input.get_next.remote())

        self.order_buffer = self.comparator(self.order_buffer, self.ASC)
        return self.order_buffer

    def execute(self, input_tuples, span_ctx):
        """
        input_tuples (List[]): A list of Atuples coming from the GroupBy operator.
        span_ctx(SpanContext Object): Span context from the parent span of "OrderBy" operator
        """
        tracer = self.init_tracer(self.jaeger_name)
        with tracer.start_span(operation_name=self.name, references=opentracing.child_of(span_ctx)) as span:
            span.set_tag(key="pid", value=os.getpid())

            if input_tuples is not None:
                self.execute_order_buffer.extend(input_tuples)
            else:
                self.num_finished_GroupBy += 1
                if self.num_finished_GroupBy == 2:
                    bucket_handler = sorted(self.execute_order_buffer, key=lambda x: x.tuple[1], reverse=self.ASC)
                    # Call "TopK" handler
                    self.handler.execute.remote(bucket_handler, span.context)
                    self.handler.execute.remote(None, span.context)
            return True

    def set_next_handler(self, handler):
        self.handler = handler

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        input_tuples = []
        output_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t].operator
            input_tuples.append(self.inout_mappings[t])
        output_lineage.extend(next_operator.lineage(input_tuples))
        return output_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        # Since at OrderBy operator the input tuple is the same with the output tuple we bypass the inout_attr_mappings
        # dict at this operator and redirect the backtracking to same att_index as input
        input_tuples = []
        output_where_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t].operator
            input_tuples.append(self.inout_mappings[t])
        output_where_lineage.extend(next_operator.where(att_index, input_tuples))
        return output_where_lineage


# Top-k operator
@ray.remote
class TopK(Operator):
    """TopK operator.

    Attributes:
        input (Operator): A handle to the input.
        k (int): The maximum number of tuples to output.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    # Initializes top-k operator
    def __init__(self, input, k, track_prov, propagate_prov):
        Operator.__init__(self, name="TopK", track_prov=track_prov,
                          propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        # k starts from 0
        self.k = k
        self.topk_buffer = []
        self.execute_topk_buffer = []
        self.done = 0
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.inout_mappings = {}

    # Returns the first k tuples in the input (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        output_tuples = []
        dict_metadata = {}
        # Check if we have assigned value to self.topK_buffer to guarantee that it is done
        if self.topk_buffer:
            return None

        input_tuples = ray.get(self.input.get_next.remote())
        # While input has not been exhausted keep adding to topk_buffer
        while input_tuples is not None:
            for t in input_tuples:
                self.topk_buffer.append(t)
            input_tuples = ray.get(self.input.get_next.remote())

        for i, t in enumerate(self.topk_buffer):
            if self.propagate_prov:
                dict_metadata = t.metadata
                # Useful only for TaskIV
                dict_metadata.update({"OrderByOutput": self.topk_buffer})
            generated_tuple = ATuple(t.tuple, metadata=dict_metadata if self.propagate_prov else None, operator=self)
            if self.track_prov:
                self.inout_mappings.update({generated_tuple: t})
            output_tuples.append(generated_tuple)
            # Check the limit k
            if i == self.k:
                return output_tuples

    def execute(self, input_tuples, span_ctx):
        """
        input_tuples (List[]): A list of Atuples coming from the OrderBY operator.
        span_ctx(SpanContext Object): Span context from the parent span of "TopK" operator
        """
        tracer = self.init_tracer(self.jaeger_name)
        with tracer.start_span(operation_name=self.name, references=opentracing.child_of(span_ctx)) as span:
            span.set_tag(key="pid", value=os.getpid())
            bucket_handler = []

            if input_tuples is not None:
                for t in input_tuples:
                    self.execute_topk_buffer.append(t)
                return True
            else:
                for i, t in enumerate(self.execute_topk_buffer):
                    bucket_handler.append(t)
                    # Check the limit k
                    if i == self.k:
                        # Call "Project" handler
                        self.handler.execute.remote(bucket_handler,span.context)
                        self.handler.execute.remote(None,span.context)
                        return True

    def set_next_handler(self, handler):
        self.handler = handler

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        input_tuples = []
        output_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t].operator
            input_tuples.append(self.inout_mappings[t])
        output_lineage.extend(next_operator.lineage(input_tuples))
        return output_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        # Since at TopK operator the input tuple is the same with the output tuple we bypass the inout_attr_mappings
        # dict at this operator and redirect the backtracking to same att_index as input
        input_tuples = []
        output_where_lineage = []
        for t in tuples:
            next_operator = self.inout_mappings[t].operator
            input_tuples.append(self.inout_mappings[t])
        output_where_lineage.extend(next_operator.where(att_index, input_tuples))
        return output_where_lineage


# Filter operator
@ray.remote
class Select(Operator):
    """Select operator.

    Attributes:
        input (Operator): A handle to the input.
        predicate (function): The selection predicate.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """

    # Initializes select operator
    def __init__(self, input, predicate, batch_size, track_prov,
                 propagate_prov):
        Operator.__init__(self, name="Select", track_prov=track_prov,
                          propagate_prov=propagate_prov)
        # YOUR CODE HERE
        self.input = input
        self.predicate = predicate
        self.selection_buffer = []
        self.batch_size = batch_size
        self.propagate_prov = propagate_prov
        self.track_prov = track_prov
        self.dict_metadata = {}
        self.buffer_handler_0 = []
        self.buffer_handler_1 = []
        self.num_finished_scans = 0

    # Returns next batch of tuples that pass the filter (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        output_tuples = []
        counter_batch = 0

        # Check if selection buffer is empty and if not output its content
        if self.selection_buffer:
            for i, t in enumerate(self.selection_buffer):
                # Add projection tuple to output
                output_tuples.append(t)
                counter_batch += 1
                if counter_batch == self.batch_size:
                    # Remove as many selection tuples added to output_tuples
                    self.selection_buffer = self.selection_buffer[counter_batch:]
                    return output_tuples

            # Remove as many selection tuples added to output_tuples
            self.selection_buffer = self.selection_buffer[counter_batch:]

        input_tuples = ray.get(self.input.get_next.remote())
        # Check if input has been exhausted
        if input_tuples is None:
            return None

        for t in input_tuples:
            if self.predicate(t):
                t.operator = self if self.track_prov else None
                if not self.propagate_prov:
                    t.metadata = None

                # Write input tuples to selection buffer to read them in next call of get_next()
                if counter_batch > self.batch_size:
                    self.selection_buffer.append(t)
                # Write input tuples to output_tuples
                else:
                    output_tuples.append(t)
                counter_batch += 1

        return output_tuples

    def execute(self, input_tuples, file_id, span_ctx):
        """
        input_tuples (List[]): A list of Atuples coming from the Scan operator.
        file_id (int): Identifier depicting if the input_tuples come
                       from bucket_handler_0(mod UID2 = 0) or bucket_handler_1(mod UID2 = 1)
        span_ctx(SpanContext Object): Span context from the parent span of "Select" operator
        """
        tracer = self.init_tracer(self.jaeger_name)
        with tracer.start_span(operation_name=self.name, references=opentracing.child_of(span_ctx)) as span:
            span.set_tag(key="pid", value=os.getpid())

            bucket_handler_0 = []
            bucket_handler_1 = []
            counter_batch_0 = 0
            counter_batch_1 = 0

            # Check if buffer_bucket_0 is empty and if not output its content
            if self.buffer_handler_0:
                for i, t in enumerate(self.buffer_handler_0):
                    # Add projection tuple to output
                    bucket_handler_0.append(t)
                    counter_batch_0 += 1
                    if counter_batch_0 == self.batch_size:
                        self.handlers[0].execute.remote(bucket_handler_0, file_id, span.context)
                        # Remove as many selection tuples added to bucket_handler_0
                        self.buffer_handler_0 = self.buffer_handler_0[counter_batch_0:]

                # Remove as many selection tuples added to bucket_handler
                self.buffer_handler_0 = self.buffer_handler_0[counter_batch_0:]

            # Check if buffer_handler_1 is empty and if not output its content
            if self.buffer_handler_1:
                for i, t in enumerate(self.buffer_handler_1):
                    # Add projection tuple to output
                    bucket_handler_1.append(t)
                    counter_batch_1 += 1
                    if counter_batch_1 == self.batch_size:
                        self.handlers[1].execute.remote(bucket_handler_1, file_id, span.context)
                        # Remove as many selection tuples added to bucket_handler_1
                        self.buffer_handler_1 = self.buffer_handler_1[counter_batch_1:]

                # Remove as many selection tuples added to bucket_handler
                self.buffer_handler_1 = self.buffer_handler_1[counter_batch_1:]

            # Check if input has been exhausted
            if input_tuples is None:
                self.num_finished_scans += 1
                # Send last tuples stored at bucket_handler
                if counter_batch_0 > 0:
                    self.handlers[0].execute.remote(bucket_handler_0, file_id, span.context)
                # Send last tuples stored at bucket_handler
                if counter_batch_1 > 0:
                    self.handlers[1].execute.remote(bucket_handler_1, file_id, span.context)
                # Check that all Scan Friends table have been finished and
                # send None to indicate that input has been exhausted
                if self.num_finished_scans == 4:
                    self.handlers[0].execute.remote(None, file_id, span.context)
                    self.handlers[1].execute.remote(None, file_id, span.context)
            else:
                bucket_handler_0 = []
                bucket_handler_1 = []
                counter_batch_0 = 0
                counter_batch_1 = 0
                for t in input_tuples:
                    if self.predicate(t):
                        if file_id == 0:
                            # Write input tuples to buffer_handler_0 to read them in next call of execute()
                            if counter_batch_0 > self.batch_size:
                                self.buffer_handler_0.append(t)
                                counter_batch_0 += 1
                            # Write input tuples to bucket_handler_0
                            else:
                                bucket_handler_0.append(t)
                        else:
                            # Write input tuples to buffer_handler_1 to read them in next call of execute()
                            if counter_batch_1 > self.batch_size:
                                self.buffer_handler_1.append(t)
                                counter_batch_1 += 1
                            # Write input tuples to bucket_handler_1
                            else:
                                bucket_handler_1.append(t)
                # Call "Join" handlers
                if bucket_handler_0:
                    self.handlers[0].execute.remote(bucket_handler_0, file_id, span.context)
                if bucket_handler_1:
                    self.handlers[1].execute.remote(bucket_handler_1, file_id, span.context)
        return True

    def set_next_handlers(self, handlers):
        self.handlers = handlers

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        input_tuples = []
        output_lineage = []

        for t in tuples:
            # Scan has no mapping to refer since it is the tuple itself
            if self.input.name == 'Scan':
                input_tuples.append(t)
            else:
                input_tuples.append(self.input.inout_mappings[t])
        # Since at Select operator the input tuple is the same with the output tuple we bypass
        # the inout_mappings dict at this operator and redirect the backtracking to our previous operator self.input
        next_operator = self.input
        output_lineage.extend(next_operator.lineage(input_tuples))
        return output_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        input_tuples = []
        output_where_lineage = []
        for t in tuples:
            # Scan has no mapping to refer since it is the tuple itself
            if self.input.name == 'Scan':
                input_tuples.append(t)
            else:
                input_tuples.append(self.input.inout_mappings[t])
        # Since at Select operator the input tuple is the same with the output tuple we bypass the inout_mappings dict
        # at this operator and redirect the backtracking to our previous operator self.input
        next_operator = self.input
        output_where_lineage.extend(next_operator.where(att_index, input_tuples))
        return output_where_lineage


# Returns the average of values of the distinct key
def AVG(input_tuples, key):
    sum = 0
    for t in input_tuples:
        sum += int(t.tuple[key])
    return sum / len(input_tuples)


# Returns a True/False in a response to induct a tuple in the selection operator
def select_predicate_friends(t):
    if int(t.tuple[0]) == args.uid:
        return True
    return False


# Returns a True/False in a response to induct a tuple in the selection operator
def select_predicate_movie_ratings(t):
    if int(t.tuple[1]) == args.mid:
        return True
    return False


# Implement quicksort for Order By operator
def comparator(input_tuples, order):
    less = []
    equal = []
    greater = []

    if len(input_tuples) > 1:
        pivot = float(input_tuples[0].tuple[1])
        for t in input_tuples:
            if float(t.tuple[1]) < pivot:
                less.append(t)
            elif float(t.tuple[1]) == pivot:
                equal.append(t)
            elif float(t.tuple[1]) > pivot:
                greater.append(t)
        if order:
            return comparator(less, order) + equal + comparator(greater, order)
        else:
            return comparator(greater, order) + equal + comparator(less, order)
    else:
        return input_tuples


if __name__ == "__main__":

    # TASK 1: Implement 'likeness' prediction query for User A and Movie M
    #
    # SELECT AVG(R.Rating)
    # FROM Friends as F, Ratings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'

    # YOUR CODE HERE

    # Parse the arguments
    args = parseArguments()
    if args.assignment == 1:
        logger.info("Assignment #1")
        if args.task == 1:
            scan_friends = Scan(filepath=args.friends, filter=select_predicate_friends, track_prov=False,
                                propagate_prov=False, batch_size=100000)
            scan_movie_ratings = Scan(filepath=args.ratings, filter=select_predicate_movie_ratings, track_prov=False,
                                      propagate_prov=False,
                                      batch_size=100000)
            join_data = Join(left_input=scan_friends, right_input=scan_movie_ratings, left_join_attribute=1,
                             right_join_attribute=0, track_prov=False, propagate_prov=False, batch_size=100000)

            join_tuples = []
            tuples = join_data.get_next()
            while tuples is not None:
                join_tuples.extend(tuples)
                tuples = join_data.get_next()
            logger.info(round(AVG(join_tuples, key=3), 2))

        # TASK 2: Implement recommendation query for User A
        #
        # SELECT R.MID
        # FROM ( SELECT R.MID, AVG(R.Rating) as score
        #        FROM Friends as F, Ratings as R
        #        WHERE F.UID2 = R.UID
        #              AND F.UID1 = 'A'
        #        GROUP BY R.MID
        #        ORDER BY score DESC
        #        LIMIT 1 )

        # YOUR CODE HERE
        if args.task == 2:

            scan_friends = Scan(filepath=args.friends, filter=select_predicate_friends, track_prov=False,
                                propagate_prov=False, batch_size=100000)
            scan_movie_ratings = Scan(filepath=args.ratings, filter=None, track_prov=False, propagate_prov=False,
                                      batch_size=100000)
            join_data = Join(left_input=scan_friends, right_input=scan_movie_ratings, left_join_attribute=1,
                             right_join_attribute=0, track_prov=False, propagate_prov=False, batch_size=100000)
            groupBy_data = GroupBy(input=join_data, key=2, value=3, track_prov=False, propagate_prov=False, agg_fun=AVG)
            order_data = OrderBy(input=groupBy_data, comparator=comparator, track_prov=False, propagate_prov=False,
                                 ASC=False)
            topK_data = TopK(input=order_data, track_prov=False, propagate_prov=False, k=0)
            projection_data = Project(input=topK_data, fields_to_keep=[0], track_prov=False, propagate_prov=False,
                                      batch_size=100000)

            tuples = projection_data.get_next()
            output_tuples = []
            while tuples is not None:
                for t in tuples:
                    logger.info(t.tuple)
                tuples = projection_data.get_next()

        # TASK 3: Implement explanation query for User A and Movie M
        #
        # SELECT HIST(R.Rating) as explanation
        # FROM Friends as F, Ratings as R
        # WHERE F.UID2 = R.UID
        #       AND F.UID1 = 'A'
        #       AND R.MID = 'M'

        # YOUR CODE HERE
        if args.task == 3:

            scan_friends = Scan(filepath=args.friends, filter=select_predicate_friends, track_prov=False,
                                propagate_prov=False, batch_size=100000)
            scan_movie_ratings = Scan(filepath=args.ratings, filter=None, track_prov=False, propagate_prov=False,
                                      batch_size=100000)
            join_data = Join(left_input=scan_friends, right_input=scan_movie_ratings, left_join_attribute=1,
                             right_join_attribute=0, track_prov=False, propagate_prov=False, batch_size=100000)
            histogram_data = Histogram(input=join_data, track_prov=False, propagate_prov=False, key=3)

            tuples = histogram_data.get_next()
            while tuples is not None:
                for t in tuples:
                    logger.info(t.tuple)
                tuples = histogram_data.get_next()

        # TASK 4: Turn your data operators into Ray actors
        #
        # NOTE (john): Add your changes for Task 4 to a new git branch 'ray'

    if args.assignment == 2:
        logger.info("Assignment #2")
        scan_friends = Scan(filepath=args.friends, filter=select_predicate_friends, track_prov=True,
                            propagate_prov=True, batch_size=2)
        scan_movie_ratings = Scan(filepath=args.ratings, filter=None, track_prov=True, propagate_prov=True,
                                  batch_size=2)
        join_data = Join(left_input=scan_friends, right_input=scan_movie_ratings, left_join_attribute=1,
                         right_join_attribute=0, track_prov=True, propagate_prov=True, batch_size=2)
        groupBy_data = GroupBy(input=join_data, key=2, value=3, track_prov=True, propagate_prov=True, agg_fun=AVG)
        order_data = OrderBy(input=groupBy_data, comparator=comparator, track_prov=True, propagate_prov=True,
                             ASC=False)
        topK_data = TopK(input=order_data, track_prov=True, propagate_prov=True, k=0)
        projection_data = Project(input=topK_data, fields_to_keep=[0], track_prov=True, propagate_prov=True,
                                  batch_size=2)

        tuples = projection_data.get_next()
        recommendation = []
        while tuples is not None:
            for t in tuples:
                recommendation.append(t)
            tuples = projection_data.get_next()

        # TASK 1: Implement lineage query for movie recommendation
        # YOUR CODE HERE
        if args.task == 1:
            logger.info(recommendation[0].lineage())

        # TASK 2: Implement where-provenance query for 'likeness' prediction
        # YOUR CODE HERE
        if args.task == 2:
            logger.info(recommendation[0].where(0))

        # TASK 3: Implement how-provenance query for movie recommendation
        # YOUR CODE HERE
        if args.task == 3:
            logger.info(recommendation[0].how())

        # TASK 4: Retrieve most responsible tuples for movie recommendation
        # YOUR CODE HERE
        if args.task == 4:
            logger.info(recommendation[0].responsible_inputs())

    if args.assignment == 4:
        logger.info("Assignment #4")
        logger.info("Ray initialization")
        # ray.init(local_mode=True)
        ray.init()
        if args.task == 1:

            # TASK 1: Implement recommendation query for User A
            #
            # SELECT R.MID
            # FROM ( SELECT R.MID, AVG(R.Rating) as score
            #        FROM Friends as F, Ratings as R
            #        WHERE F.UID2 = R.UID
            #              AND F.UID1 = 'A'
            #        GROUP BY R.MID
            #        ORDER BY score DESC
            #        LIMIT 1 )

            scan_friends = Scan.remote(filepath=args.friends, filter=select_predicate_friends, batch_size=10000,
                                       track_prov=False, propagate_prov=False)
            scan_movie_ratings = Scan.remote(filepath=args.ratings, filter=None, batch_size=10000, track_prov=False,
                                             propagate_prov=False)
            join_data = Join.remote(left_input=scan_friends, right_input=scan_movie_ratings, left_join_attribute=1,
                                    right_join_attribute=0, batch_size=10000, track_prov=False, propagate_prov=False)
            groupBy_data = GroupBy.remote(input=join_data, key=2, value=3, agg_fun=AVG, track_prov=False,
                                          propagate_prov=False)
            order_data = OrderBy.remote(input=groupBy_data, comparator=comparator, ASC=False, track_prov=False,
                                        propagate_prov=False)
            topK_data = TopK.remote(input=order_data, k=0, track_prov=False, propagate_prov=False)
            projection_data = Project.remote(input=topK_data, fields_to_keep=[0], batch_size=10000, track_prov=False,
                                             propagate_prov=False)

            tuples = ray.get(projection_data.get_next.remote())
            while tuples is not None:
                for t in tuples:
                    logger.info(t.tuple)
                tuples = ray.get(projection_data.get_next.remote())

        if args.task == 2:
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
            scan_friends_handler_0 = Scan.remote(filepath="../data/Assignment_4_data/friends_01.txt",
                                                 filter=None,
                                                 batch_size=10000,
                                                 track_prov=False, propagate_prov=False)
            scan_friends_handler_0.set_next_handlers.remote([select_handler_0, select_handler_1])
            scan_friends_handler_1 = Scan.remote(filepath="../data/Assignment_4_data/friends_02.txt",
                                                 filter=None,
                                                 batch_size=10000,
                                                 track_prov=False, propagate_prov=False)
            scan_friends_handler_1.set_next_handlers.remote([select_handler_0, select_handler_1])
            scan_friends_handler_2 = Scan.remote(filepath="../data/Assignment_4_data/friends_03.txt",
                                                 filter=None,
                                                 batch_size=10000,
                                                 track_prov=False, propagate_prov=False)
            scan_friends_handler_2.set_next_handlers.remote([select_handler_0, select_handler_1])
            scan_friends_handler_3 = Scan.remote(filepath="../data/Assignment_4_data/friends_04.txt",
                                                 filter=None,
                                                 batch_size=10000,
                                                 track_prov=False, propagate_prov=False)
            scan_friends_handler_3.set_next_handlers.remote([select_handler_0, select_handler_1])

            # Scan ratings operator instantiation
            scan_ratings_handler_0 = Scan.remote(filepath="../data/Assignment_4_data/ratings_01.txt",
                                                 filter=None, batch_size=10000, track_prov=False, propagate_prov=False)
            scan_ratings_handler_0.set_next_handlers.remote([join_handler_0, join_handler_1])
            scan_ratings_handler_1 = Scan.remote(filepath="../data/Assignment_4_data/ratings_02.txt",
                                                 filter=None, batch_size=10000, track_prov=False, propagate_prov=False)
            scan_ratings_handler_1.set_next_handlers.remote([join_handler_0, join_handler_1])
            scan_ratings_handler_2 = Scan.remote(filepath="../data/Assignment_4_data/ratings_03.txt",
                                                 filter=None, batch_size=10000, track_prov=False, propagate_prov=False)
            scan_ratings_handler_2.set_next_handlers.remote([join_handler_0, join_handler_1])
            scan_ratings_handler_3 = Scan.remote(filepath="../data/Assignment_4_data/ratings_04.txt",
                                                 filter=None, batch_size=10000, track_prov=False, propagate_prov=False)
            scan_ratings_handler_3.set_next_handlers.remote([join_handler_0, join_handler_1])

            # Start query execution
            print("Start scanning")
            scan_ratings_handler_0.execute.remote()
            scan_ratings_handler_1.execute.remote()
            scan_ratings_handler_2.execute.remote()
            scan_ratings_handler_3.execute.remote()

            scan_friends_handler_0.execute.remote()
            scan_friends_handler_1.execute.remote()
            scan_friends_handler_2.execute.remote()
            scan_friends_handler_3.execute.remote()
            print("Finish scanning")

            # Blocking call
            time.sleep(30)
            movie_id = ray.get(sink_handler.get_result.remote())
            print(movie_id)
