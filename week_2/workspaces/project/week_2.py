from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    String,
    graph,
    op,
)
from workspaces.config import REDIS, S3, S3_FILE
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    out=Out(dagster_type=List[Stock]),
    required_resource_keys={"s3"},
    tags={"kind": "s3"},
)
def get_s3_data(context: OpExecutionContext):
    # This op reads a file from S3 (provided as a config schema) and converts the contents into a list of our custom data type Stock.
    # Last week we used the csv module to read the contents of a local file and return an iterator.
    # We will replace that functionality with our S3 resource and use the S3 client method get_data to read the contents a file from remote storage
    # (in this case our localstack version of S3 within Docker).

    # get S3 key from config
    key = context.op_config["s3_key"]

    # get data from S3
    result = context.resources.s3.get_data(key)

    stocks = []

    # iterate through S3 data and append to stocks with Stock class
    for i in range(len(result)):
        stocks.append(Stock.from_list(result[i]))

    return stocks


@op(
    ins={"stocks": In(dagster_type=List[Stock], description="List of Stocks")},
    out={
        "aggregation": Out(dagster_type=Aggregation, description="Highest value of stock")
    }
)
def process_data(stocks):
    highest_stock = max(stocks, key = lambda x: x.high)

    return Aggregation(date=highest_stock.date, high=highest_stock.high)

    
@op(
    ins={"aggregation": In(dagster_type=Aggregation, description="Highest value stock")},
    out=Out(Nothing),
    required_resource_keys={"redis"},
    tags={"kind": "redis"},
)
def put_redis_data(context: OpExecutionContext, aggregation: Aggregation):
    # This op relies on the redis_resource. In week one, our op did not do anything besides accept the output from the processing app.
    # Now we want to take that output (our Aggregation custom type) and upload it to Redis. 
    # Luckily, our wrapped Redis client has a method to do just that.
    # If you look at the put_data method, it takes in a name and a value and uploads them to our cache.
    # Our Aggregation types has two properties to it, a date and a high.
    # The date should be the name and the high should be our value, but be careful because the put_data method 
    # expects those values as strings.

    date_str = str(aggregation.date)
    high_str = str(aggregation.high)

    # put data into Redis
    context.resources.redis.put_data(
        name=date_str,
        value=high_str
    )


@op(
    ins={"aggregation": In(dagster_type=Aggregation)},
    required_resource_keys={"s3"},
    tags={"kind": "s3"},
    description="Upload an Aggregation to S3 file",
)
def put_s3_data(context: OpExecutionContext, aggregation: Aggregation):
    # This op also relies on the same S3 resource as get_s3_data.
    # For the sake of this project we will use the same bucket so we can leverage the same configuration.
    # As with the redis op we will take in the aggregation from and write to it into a file in S3.
    # The key name for this file should not be set in a config (as it is with the get_s3_data op) but should be generated within the op itself.

    # get date from Aggregation
    aggregation_dt = aggregation.date.strftime('%Y_%m_%d')

    s3_key = f"/aggregations/{aggregation_dt}.csv"
    context.resources.s3.put_data(
        key_name=s3_key,
        data=aggregation,
    )


@graph
def machine_learning_graph():
    stocks = get_s3_data()

    highest_stock = process_data(stocks) # get stock with greatest high price

    put_redis_data(highest_stock)

    put_s3_data(highest_stock)



local = {
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
)
