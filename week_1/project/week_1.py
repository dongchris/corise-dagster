import csv
from datetime import datetime
from typing import Iterator, List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    String,
    job,
    op,
    usable_as_dagster_type,
)
from pydantic import BaseModel


@usable_as_dagster_type(description="Stock data")
class Stock(BaseModel):
    date: datetime
    close: float
    volume: int
    open: float
    high: float
    low: float

    @classmethod
    def from_list(cls, input_list: List[str]):
        """Do not worry about this class method for now"""
        return cls(
            date=datetime.strptime(input_list[0], "%Y/%m/%d"),
            close=float(input_list[1]),
            volume=int(float(input_list[2])),
            open=float(input_list[3]),
            high=float(input_list[4]),
            low=float(input_list[5]),
        )


@usable_as_dagster_type(description="Aggregation of stock data")
class Aggregation(BaseModel):
    date: datetime
    high: float


def csv_helper(file_name: str) -> Iterator[Stock]:
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield Stock.from_list(row)

@op(
    config_schema={"s3_key": String},
    out=Out(dagster_type=List[Stock])
)
def get_s3_data_op(context):
    # This op will bring in data and process it into our custom data type.
    # Since this is the first op in our DAG the input will not be from another op but will be provided via the config_schema.
    # This config schema will take in one parameter, a string name s3_key. The output of the op is a list of Stock.

    # For week 1 we want the focus to be on the specifics of Dagster so a helper function provider csv_helper
    # which takes in a file name and yields a generator of Stock using the class method for our custom data type.

    file_name = context.op_config["s3_key"]

    result = list(csv_helper(file_name))

    return result


@op(
    ins={"stocks": In(dagster_type=List[Stock], description="List of Stocks")},
    out={
        "aggregation": Out(dagster_type=Aggregation, description="Highest value of stock")
    }
)
def process_data_op(stocks):
    # This op will require the output of the get_s3_data (which will be a list of Stock).
    # The output of the process_data will be our custom type Aggregation

    # The processing occurring within the op will take the list of stocks and determine the Stock with the greatest high value.

    greatest_high_value = 0
    stock_result = None
    for stock in stocks:
        if stock.high > greatest_high_value:
            stock_result = Aggregation(date=stock.date, high=stock.high)
            greatest_high_value = stock.high
    return stock_result


@op(
    ins={"aggregation": In(dagster_type=Aggregation, description="Highest value stock")},
    out=Out(Nothing)
)
def put_redis_data_op(aggregation):
    # For now, this op will be doing very little (it is fine if the function body remains just pass).
    # However, it will need to accept the Aggregation type from your process_data op.

    pass


@op(
    ins={"aggregation": In(dagster_type=Aggregation, description="Highest value stock")},
    out=Out(Nothing)
)
def put_s3_data_op(aggregation):
    pass

@job
def machine_learning_job():
    stocks = get_s3_data_op() # get s3 data based on context

    highest_stock = process_data_op(stocks) # get stock with greatest high price

    put_redis_data_op(highest_stock)

    put_s3_data_op(highest_stock)
