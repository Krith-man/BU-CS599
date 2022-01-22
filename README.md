# This repository contains solutions to the CS599 L1 assignments 

## Prerequisites for running queries:

1. Python (3.7+)
2. [Pytest](https://docs.pytest.org/en/stable/)
3. [Ray](https://ray.io)

## Input Data

Queries of assignments 1 and 2 expect two space-delimited text files (similar to CSV files). 

The first file (friends) must include records of the form:

|UID1 (int)|UID2 (int)|
|----|----|
|1   |2342|
|231 |3   |
|... |... |

The second file (ratings) must include records of the form:

|UID (int)|MID (int)|RATING (int)|
|---|---|------|
|1  |10 |4     |
|231|54 |2     |
|...|...|...   |

Queries of "tests.py" script expect two space-delimited text files (similar to CSV files). 

The first file (Countries) must include records of the form:

|UID (int)|Country (string)|Country_Rating (int)|
|---|---|------|
|1  |Italy |5     |
|2|France |4     |
|...|...|...   |

The second file (Towns) must include records of the form:

|UID (int)|Town (string)|Town_Rating (int)|
|---|---|------|
|1  |Rome |5     |
|1|Milano |4     |
|...|...|...   |

## Running queries of Assignment 1

You can run queries as shown below: 

```bash
$ python assignment_12.py --assignment 1 --task [task_number] --friends [path_to_friends_file.txt] --ratings [path_to_ratings_file.txt] --uid [user_id] --mid [movie_id]
```

For example, the following command runs the 'likeness prediction' query of the first task for user id 10 and movie id 3:

```bash
$ python assignment_12.py --assignment 1 --task 1 --friends ../data/Assignment_1_data/Friends.txt  --ratings ../data/Assignment_1_data/Ratings.txt --uid 10 --mid 3
```

The 'recommendation' query of the second task does not require a movie id. If you provide a `--mid` argument, it will be simply ignored.

Also, you are able to run the tests.py script with the usage of pytest package as shown below:

```bash
$ pytest tests.py
```
## Running queries of Assignment 2

You can run queries as shown below: 

```bash
$ python assignment_12.py --assignment 2 --task [task_number] --friends [path_to_friends_file.txt] --ratings [path_to_ratings_file.txt] --uid [user_id] --mid [movie_id]
```

Since Task I, II and & share the same data while Task IV has its own data we diversify their data paths. 
Therefore, running task I,II and III you only have to change task number in the below command:
```bash
$ python assignment_12.py --assignment 2 --task 1 --friends ../data/Assignment_2_data/Task_I_II_III/Friends.txt  --ratings ../data/Assignment_2_data/Task_I_II_III/Ratings.txt --uid 0 --mid 3
```
While for running task IV you have to run the following command:
```bash
$ python assignment_12.py --assignment 2 --task 4 --friends ../data/Assignment_2_data/Task_IV/Friends.txt  --ratings ../data/Assignment_2_data/Task_IV/Ratings.txt --uid 0 --mid 3
```

Also, you are able to run the tests.py script with the usage of pytest package as shown below:

```bash
$ pytest tests.py
```

## Running queries of Assignment 3

You can run queries as shown below: 

```bash
$ python explain_pipeline.py
```

The results for Task I are stored at "../data/Assignment_3_data/ADNI3/output/classified_AD.txt" <br />
The results for Task II are stored at "../data/Assignment_3_data/ADNI3/output/SHAP/data/" <br />
The results for Task III are stored at "../data/Assignment_3_data/ADNI3/output/SHAP/heatmaps/" <br />
The results for Task IV are stored at "../data/Assignment_3_data/ADNI3/output/top5/"

Also, you are able to run the tests.py script with the usage of pytest package as shown below:

```bash
$ pytest tests.py
```

The results for testing are stored in "../data/Assignment_3_data/ADNI3/testing/"

## Running queries of Assignment 4

You can run the Task II as shown below: 

```bash
$ python assignment_4.py --assignment 4 --task 2 --uid 0 --mid 3
```

The data for Task II are stored at "../data/Assignment_4_data". <br />
The screenshots for Task III & IV are stored in the directory "Jaeger_traces".

Also, you are able to run the tests.py script with the usage of pytest package as shown below:

```bash
$ pytest tests.py
```

The data for testing are stored in "../data/Assignment_4_testing_data".
