# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name

import time
from datetime import date
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession

from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.transforms import BucketTransform, IdentityTransform, MonthTransform
from pyiceberg.types import (
    BooleanType,
    DateType,
    IntegerType,
    NestedField,
    StringType,
)

TABLE_SCHEMA = Schema(
    NestedField(field_id=1, name="foo", field_type=BooleanType(), required=False),
    NestedField(field_id=2, name="bar", field_type=StringType(), required=False),
    NestedField(field_id=4, name="baz", field_type=IntegerType(), required=False),
    NestedField(field_id=10, name="qux", field_type=DateType(), required=False),
)

ARROW_SCHEMA = pa.schema([
    ("foo", pa.bool_()),
    ("bar", pa.string()),
    ("baz", pa.int32()),
    ("qux", pa.date32()),
])

ARROW_TABLE = pa.Table.from_pylist(
    [
        {
            "foo": True,
            "bar": "bar_string",
            "baz": 123,
            "qux": date(2024, 3, 7),
        }
    ],
    schema=ARROW_SCHEMA,
)

ARROW_SCHEMA_WITH_IDS = pa.schema([
    pa.field("foo", pa.bool_(), nullable=False, metadata={"PARQUET:field_id": "1"}),
    pa.field("bar", pa.string(), nullable=False, metadata={"PARQUET:field_id": "2"}),
    pa.field("baz", pa.int32(), nullable=False, metadata={"PARQUET:field_id": "3"}),
    pa.field("qux", pa.date32(), nullable=False, metadata={"PARQUET:field_id": "4"}),
])


ARROW_TABLE_WITH_IDS = pa.Table.from_pylist(
    [
        {
            "foo": True,
            "bar": "bar_string",
            "baz": 123,
            "qux": date(2024, 3, 7),
        }
    ],
    schema=ARROW_SCHEMA_WITH_IDS,
)

ARROW_SCHEMA_UPDATED = pa.schema([
    ("foo", pa.bool_()),
    ("baz", pa.int32()),
    ("qux", pa.date32()),
    ("quux", pa.int32()),
])

ARROW_TABLE_UPDATED = pa.Table.from_pylist(
    [
        {
            "foo": True,
            "baz": 123,
            "qux": date(2024, 3, 7),
            "quux": 234,
        }
    ],
    schema=ARROW_SCHEMA_UPDATED,
)


def _create_table(
    session_catalog: Catalog, identifier: str, format_version: int, partition_spec: Optional[PartitionSpec] = None
) -> Table:
    try:
        session_catalog.drop_table(identifier=identifier)
    except NoSuchTableError:
        pass

    tbl = session_catalog.create_table(
        identifier=identifier,
        schema=TABLE_SCHEMA,
        properties={"format-version": str(format_version)},
        partition_spec=partition_spec if partition_spec else PartitionSpec(),
    )

    return tbl


@pytest.fixture(name="format_version", params=[pytest.param(1, id="format_version=1"), pytest.param(2, id="format_version=2")])
def format_version_fixure(request: pytest.FixtureRequest) -> Iterator[int]:
    """Fixture to run tests with different table format versions."""
    yield request.param


@pytest.mark.integration
def test_add_files_to_unpartitioned_table(spark: SparkSession, session_catalog: Catalog, format_version: int) -> None:
    identifier = f"default.unpartitioned_table_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    file_paths = [f"s3://warehouse/default/unpartitioned/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(ARROW_TABLE)

    # add the parquet files as data files
    tbl.add_files(file_paths=file_paths)

    # NameMapping must have been set to enable reads
    assert tbl.name_mapping() is not None

    rows = spark.sql(
        f"""
        SELECT added_data_files_count, existing_data_files_count, deleted_data_files_count
        FROM {identifier}.all_manifests
    """
    ).collect()

    assert [row.added_data_files_count for row in rows] == [5]
    assert [row.existing_data_files_count for row in rows] == [0]
    assert [row.deleted_data_files_count for row in rows] == [0]

    df = spark.table(identifier)
    assert df.count() == 5, "Expected 5 rows"
    for col in df.columns:
        assert df.filter(df[col].isNotNull()).count() == 5, "Expected all 5 rows to be non-null"

    # check that the table can be read by pyiceberg
    assert len(tbl.scan().to_arrow()) == 5, "Expected 5 rows"


@pytest.mark.integration
def test_add_files_to_unpartitioned_table_raises_file_not_found(
    spark: SparkSession, session_catalog: Catalog, format_version: int
) -> None:
    identifier = f"default.unpartitioned_raises_not_found_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    file_paths = [f"s3://warehouse/default/unpartitioned_raises_not_found/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(ARROW_TABLE)

    # add the parquet files as data files
    with pytest.raises(FileNotFoundError):
        tbl.add_files(file_paths=file_paths + ["s3://warehouse/default/unpartitioned_raises_not_found/unknown.parquet"])


@pytest.mark.integration
def test_add_files_to_unpartitioned_table_raises_has_field_ids(
    spark: SparkSession, session_catalog: Catalog, format_version: int
) -> None:
    identifier = f"default.unpartitioned_raises_field_ids_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    file_paths = [f"s3://warehouse/default/unpartitioned_raises_field_ids/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA_WITH_IDS) as writer:
                writer.write_table(ARROW_TABLE_WITH_IDS)

    # add the parquet files as data files
    with pytest.raises(NotImplementedError):
        tbl.add_files(file_paths=file_paths)


@pytest.mark.integration
def test_add_files_to_unpartitioned_table_with_schema_updates(
    spark: SparkSession, session_catalog: Catalog, format_version: int
) -> None:
    identifier = f"default.unpartitioned_table_schema_updates_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    file_paths = [f"s3://warehouse/default/unpartitioned_schema_updates/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(ARROW_TABLE)

    # add the parquet files as data files
    tbl.add_files(file_paths=file_paths)

    # NameMapping must have been set to enable reads
    assert tbl.name_mapping() is not None

    with tbl.update_schema() as update:
        update.add_column("quux", IntegerType())
        update.delete_column("bar")

    file_path = f"s3://warehouse/default/unpartitioned_schema_updates/v{format_version}/test-6.parquet"
    # write parquet files
    fo = tbl.io.new_output(file_path)
    with fo.create(overwrite=True) as fos:
        with pq.ParquetWriter(fos, schema=ARROW_SCHEMA_UPDATED) as writer:
            writer.write_table(ARROW_TABLE_UPDATED)

    # add the parquet files as data files
    tbl.add_files(file_paths=[file_path])
    rows = spark.sql(
        f"""
        SELECT added_data_files_count, existing_data_files_count, deleted_data_files_count
        FROM {identifier}.all_manifests
    """
    ).collect()

    assert [row.added_data_files_count for row in rows] == [5, 1, 5]
    assert [row.existing_data_files_count for row in rows] == [0, 0, 0]
    assert [row.deleted_data_files_count for row in rows] == [0, 0, 0]

    df = spark.table(identifier)
    assert df.count() == 6, "Expected 6 rows"
    assert len(df.columns) == 4, "Expected 4 columns"

    for col in df.columns:
        value_count = 1 if col == "quux" else 6
        assert df.filter(df[col].isNotNull()).count() == value_count, f"Expected {value_count} rows to be non-null"

    # check that the table can be read by pyiceberg
    assert len(tbl.scan().to_arrow()) == 6, "Expected 6 rows"


@pytest.mark.integration
def test_add_files_to_partitioned_table(spark: SparkSession, session_catalog: Catalog, format_version: int) -> None:
    identifier = f"default.partitioned_table_v{format_version}"

    partition_spec = PartitionSpec(
        PartitionField(source_id=4, field_id=1000, transform=IdentityTransform(), name="baz"),
        PartitionField(source_id=10, field_id=1001, transform=MonthTransform(), name="qux_month"),
        spec_id=0,
    )

    tbl = _create_table(session_catalog, identifier, format_version, partition_spec)

    date_iter = iter([date(2024, 3, 7), date(2024, 3, 8), date(2024, 3, 16), date(2024, 3, 18), date(2024, 3, 19)])

    file_paths = [f"s3://warehouse/default/partitioned/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "foo": True,
                                "bar": "bar_string",
                                "baz": 123,
                                "qux": next(date_iter),
                            }
                        ],
                        schema=ARROW_SCHEMA,
                    )
                )

    # add the parquet files as data files
    tbl.add_files(file_paths=file_paths)

    # NameMapping must have been set to enable reads
    assert tbl.name_mapping() is not None

    rows = spark.sql(
        f"""
        SELECT added_data_files_count, existing_data_files_count, deleted_data_files_count
        FROM {identifier}.all_manifests
    """
    ).collect()

    assert [row.added_data_files_count for row in rows] == [5]
    assert [row.existing_data_files_count for row in rows] == [0]
    assert [row.deleted_data_files_count for row in rows] == [0]

    df = spark.table(identifier)
    assert df.count() == 5, "Expected 5 rows"
    for col in df.columns:
        assert df.filter(df[col].isNotNull()).count() == 5, "Expected all 5 rows to be non-null"

    partition_rows = spark.sql(
        f"""
        SELECT partition, record_count, file_count
        FROM {identifier}.partitions
    """
    ).collect()

    assert [row.record_count for row in partition_rows] == [5]
    assert [row.file_count for row in partition_rows] == [5]
    assert [(row.partition.baz, row.partition.qux_month) for row in partition_rows] == [(123, 650)]

    # check that the table can be read by pyiceberg
    assert len(tbl.scan().to_arrow()) == 5, "Expected 5 rows"


@pytest.mark.integration
def test_add_files_to_bucket_partitioned_table_fails(spark: SparkSession, session_catalog: Catalog, format_version: int) -> None:
    identifier = f"default.partitioned_table_bucket_fails_v{format_version}"

    partition_spec = PartitionSpec(
        PartitionField(source_id=4, field_id=1000, transform=BucketTransform(num_buckets=3), name="baz_bucket_3"),
        spec_id=0,
    )

    tbl = _create_table(session_catalog, identifier, format_version, partition_spec)

    int_iter = iter(range(5))

    file_paths = [f"s3://warehouse/default/partitioned_table_bucket_fails/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "foo": True,
                                "bar": "bar_string",
                                "baz": next(int_iter),
                                "qux": date(2024, 3, 7),
                            }
                        ],
                        schema=ARROW_SCHEMA,
                    )
                )

    # add the parquet files as data files
    with pytest.raises(ValueError) as exc_info:
        tbl.add_files(file_paths=file_paths)
    assert (
        "Cannot infer partition value from parquet metadata for a non-linear Partition Field: baz_bucket_3 with transform bucket[3]"
        in str(exc_info.value)
    )


@pytest.mark.integration
def test_add_files_to_partitioned_table_fails_with_lower_and_upper_mismatch(
    spark: SparkSession, session_catalog: Catalog, format_version: int
) -> None:
    identifier = f"default.partitioned_table_mismatch_fails_v{format_version}"

    partition_spec = PartitionSpec(
        PartitionField(source_id=4, field_id=1000, transform=IdentityTransform(), name="baz"),
        spec_id=0,
    )

    tbl = _create_table(session_catalog, identifier, format_version, partition_spec)

    file_paths = [f"s3://warehouse/default/partitioned_table_mismatch_fails/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "foo": True,
                                "bar": "bar_string",
                                "baz": 123,
                                "qux": date(2024, 3, 7),
                            },
                            {
                                "foo": True,
                                "bar": "bar_string",
                                "baz": 124,
                                "qux": date(2024, 3, 7),
                            },
                        ],
                        schema=ARROW_SCHEMA,
                    )
                )

    # add the parquet files as data files
    with pytest.raises(ValueError) as exc_info:
        tbl.add_files(file_paths=file_paths)
    assert (
        "Cannot infer partition value from parquet metadata as there are more than one partition values for Partition Field: baz. lower_value=123, upper_value=124"
        in str(exc_info.value)
    )


@pytest.mark.integration
def test_add_files_snapshot_properties(spark: SparkSession, session_catalog: Catalog, format_version: int) -> None:
    identifier = f"default.unpartitioned_table_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    file_paths = [f"s3://warehouse/default/unpartitioned/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(ARROW_TABLE)

    # add the parquet files as data files
    tbl.add_files(file_paths=file_paths, snapshot_properties={"snapshot_prop_a": "test_prop_a"})

    # NameMapping must have been set to enable reads
    assert tbl.name_mapping() is not None

    summary = spark.sql(f"SELECT * FROM {identifier}.snapshots;").collect()[0].summary

    assert "snapshot_prop_a" in summary
    assert summary["snapshot_prop_a"] == "test_prop_a"


@pytest.mark.integration
def test_add_files_overwrite_to_unpartitioned_table(spark: SparkSession, session_catalog: Catalog, format_version: int) -> None:
    identifier = f"default.unpartitioned_table_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    data = {
        "foo": True,
        "bar": "bar_string",
        "baz": 123,
        "qux": date(2024, 3, 7),
    }

    # creating initial snapshot
    tbl.add_files(file_paths=_create_parquet_files(tbl, [data]))

    # testing overwrite with new data files
    file_paths = _create_parquet_files(tbl, [data for _ in range(5)])
    tbl.add_files_overwrite(file_paths=file_paths)

    # NameMapping must have been set to enable reads
    assert tbl.name_mapping() is not None

    rows = spark.sql(
        f"""
        SELECT added_data_files_count, existing_data_files_count, deleted_data_files_count
        FROM {identifier}.all_manifests
    """
    ).collect()

    assert [row.added_data_files_count for row in rows] == [1, 0, 5]
    assert [row.existing_data_files_count for row in rows] == [0, 0, 0]
    assert [row.deleted_data_files_count for row in rows] == [0, 1, 0]

    df = spark.table(identifier)
    assert df.count() == 5, "Expected 5 rows"
    for col in df.columns:
        assert df.filter(df[col].isNotNull()).count() == 5, "Expected all 5 rows to be non-null"

    # check that the table can be read by pyiceberg
    assert len(tbl.scan().to_arrow()) == 5, "Expected 5 rows"

    # check history
    assert len(tbl.scan(snapshot_id=tbl.history()[0].snapshot_id).to_arrow()) == 1, "Expected 1 row"


@pytest.mark.integration
def test_add_files_overwrite_to_unpartitioned_table_raises_file_not_found(
    spark: SparkSession, session_catalog: Catalog, format_version: int
) -> None:
    identifier = f"default.unpartitioned_raises_not_found_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    file_paths = [f"s3://warehouse/default/unpartitioned_raises_not_found/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(ARROW_TABLE)

    # add the parquet files as data files
    with pytest.raises(FileNotFoundError):
        tbl.add_files_overwrite(file_paths=file_paths + ["s3://warehouse/default/unpartitioned_raises_not_found/unknown.parquet"])


@pytest.mark.integration
def test_add_files_overwrite_to_unpartitioned_table_raises_has_field_ids(
    spark: SparkSession, session_catalog: Catalog, format_version: int
) -> None:
    identifier = f"default.unpartitioned_raises_field_ids_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    file_paths = [f"s3://warehouse/default/unpartitioned_raises_field_ids/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA_WITH_IDS) as writer:
                writer.write_table(ARROW_TABLE_WITH_IDS)

    # add the parquet files as data files
    with pytest.raises(NotImplementedError):
        tbl.add_files_overwrite(file_paths=file_paths)


@pytest.mark.integration
def test_add_files_overwrite_to_unpartitioned_table_with_schema_updates(
    spark: SparkSession, session_catalog: Catalog, format_version: int
) -> None:
    identifier = f"default.unpartitioned_table_schema_updates_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    file_paths = [f"s3://warehouse/default/unpartitioned_schema_updates/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(ARROW_TABLE)

    # add the parquet files as data files
    tbl.add_files(file_paths=file_paths)

    # NameMapping must have been set to enable reads
    assert tbl.name_mapping() is not None

    with tbl.update_schema() as update:
        update.add_column("quux", IntegerType())
        update.delete_column("bar")

    file_path = f"s3://warehouse/default/unpartitioned_schema_updates/v{format_version}/test-6.parquet"
    # write parquet files
    fo = tbl.io.new_output(file_path)
    with fo.create(overwrite=True) as fos:
        with pq.ParquetWriter(fos, schema=ARROW_SCHEMA_UPDATED) as writer:
            writer.write_table(ARROW_TABLE_UPDATED)

    # add the parquet files as data files
    tbl.add_files_overwrite(file_paths=[file_path])
    rows = spark.sql(
        f"""
        SELECT added_data_files_count, existing_data_files_count, deleted_data_files_count
        FROM {identifier}.all_manifests
    """
    ).collect()

    assert [row.added_data_files_count for row in rows] == [5, 0, 1]
    assert [row.existing_data_files_count for row in rows] == [0, 0, 0]
    assert [row.deleted_data_files_count for row in rows] == [0, 5, 0]

    df = spark.table(identifier)
    assert df.count() == 1, "Expected 1 rows"
    assert len(df.columns) == 4, "Expected 4 columns"

    for col in df.columns:
        value_count = 1
        assert df.filter(df[col].isNotNull()).count() == value_count, f"Expected {value_count} rows to be non-null"

    # check that the table can be read by pyiceberg
    assert len(tbl.scan().to_arrow()) == 1, "Expected 1 rows"


def _create_parquet_files(tbl: Table, files_constents: List[Dict[str, Any]]) -> List[str]:
    uid = uuid4().hex
    time.time_ns()
    file_paths = []
    for i, contents in enumerate(files_constents):
        file_path = f"s3://warehouse/default/partitioned/test-{time.time_ns()}/test-{uid}-{i}.parquet"
        file_paths.append(file_path)
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(
                    pa.Table.from_pylist(
                        [contents],
                        schema=ARROW_SCHEMA,
                    )
                )
    return file_paths


@pytest.mark.integration
def test_add_files_overwrite_to_partitioned_table(spark: SparkSession, session_catalog: Catalog, format_version: int) -> None:
    identifier = f"default.partitioned_table_v{format_version}"

    partition_spec = PartitionSpec(
        PartitionField(source_id=4, field_id=1000, transform=IdentityTransform(), name="baz"),
        PartitionField(source_id=10, field_id=1001, transform=MonthTransform(), name="qux_month"),
        spec_id=0,
    )

    tbl = _create_table(session_catalog, identifier, format_version, partition_spec)

    files_constents = [
        {
            "foo": True,
            "bar": "bar_string",
            "baz": 123,
            "qux": i,
        }
        for i in [date(2024, 3, 7), date(2024, 3, 8), date(2024, 3, 16), date(2024, 3, 18), date(2024, 3, 19)]
    ]
    file_paths = _create_parquet_files(tbl, files_constents)

    # add the parquet files as data files
    tbl.add_files(file_paths=file_paths)

    # NameMapping must have been set to enable reads
    assert tbl.name_mapping() is not None

    rows = spark.sql(
        f"""
        SELECT added_data_files_count, existing_data_files_count, deleted_data_files_count
        FROM {identifier}.all_manifests
    """
    ).collect()

    assert [row.added_data_files_count for row in rows] == [5]
    assert [row.existing_data_files_count for row in rows] == [0]
    assert [row.deleted_data_files_count for row in rows] == [0]

    df = spark.table(identifier)
    assert df.count() == 5, "Expected 5 rows"
    for col in df.columns:
        assert df.filter(df[col].isNotNull()).count() == 5, "Expected all 5 rows to be non-null"

    partition_rows = spark.sql(
        f"""
        SELECT partition, record_count, file_count
        FROM {identifier}.partitions
    """
    ).collect()
    assert [row.record_count for row in partition_rows] == [5]
    assert [row.file_count for row in partition_rows] == [5]
    assert [(row.partition.baz, row.partition.qux_month) for row in partition_rows] == [(123, 650)]

    # check that the table can be read by pyiceberg
    assert len(tbl.scan().to_arrow()) == 5, "Expected 5 rows"

    ##

    files_constents = [
        {
            "foo": True,
            "bar": "bar_string-overwrite",
            "baz": 456,
            "qux": i,
        }
        for i in [date(2025, 3, 7), date(2025, 3, 8), date(2025, 3, 16)]
    ]
    file_paths = _create_parquet_files(tbl, files_constents)

    # add the parquet files as data files
    tbl.add_files_overwrite(file_paths=file_paths)

    # NameMapping must have been set to enable reads
    assert tbl.name_mapping() is not None

    rows = spark.sql(
        f"""
        SELECT added_data_files_count, existing_data_files_count, deleted_data_files_count
        FROM {identifier}.all_manifests
    """
    ).collect()

    assert [row.added_data_files_count for row in rows] == [5, 0, 3]
    assert [row.existing_data_files_count for row in rows] == [0, 0, 0]
    assert [row.deleted_data_files_count for row in rows] == [0, 5, 0]

    df = spark.table(identifier)
    assert df.count() == 3, "Expected 3 rows"
    for col in df.columns:
        assert df.filter(df[col].isNotNull()).count() == 3, "Expected all 3 rows to be non-null"

    partition_rows = spark.sql(
        f"""
        SELECT partition, record_count, file_count
        FROM {identifier}.partitions
    """
    ).collect()
    assert [row.record_count for row in partition_rows] == [3]
    assert [row.file_count for row in partition_rows] == [3]
    assert [(row.partition.baz, row.partition.qux_month) for row in partition_rows] == [(456, 662)]

    # check that the table can be read by pyiceberg
    assert len(tbl.scan().to_arrow()) == 3, "Expected 3 rows"

    # check history
    assert len(tbl.scan(snapshot_id=tbl.history()[0].snapshot_id).to_arrow()) == 5, "Expected 5 rows"


@pytest.mark.integration
def test_add_files_overwrite_to_bucket_partitioned_table_fails(
    spark: SparkSession, session_catalog: Catalog, format_version: int
) -> None:
    identifier = f"default.partitioned_table_bucket_fails_v{format_version}"

    partition_spec = PartitionSpec(
        PartitionField(source_id=4, field_id=1000, transform=BucketTransform(num_buckets=3), name="baz_bucket_3"),
        spec_id=0,
    )

    tbl = _create_table(session_catalog, identifier, format_version, partition_spec)

    int_iter = iter(range(5))

    file_paths = [f"s3://warehouse/default/partitioned_table_bucket_fails/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "foo": True,
                                "bar": "bar_string",
                                "baz": next(int_iter),
                                "qux": date(2024, 3, 7),
                            }
                        ],
                        schema=ARROW_SCHEMA,
                    )
                )

    # add the parquet files as data files
    with pytest.raises(ValueError) as exc_info:
        tbl.add_files_overwrite(file_paths=file_paths)
    assert (
        "Cannot infer partition value from parquet metadata for a non-linear Partition Field: baz_bucket_3 with transform bucket[3]"
        in str(exc_info.value)
    )


@pytest.mark.integration
def test_add_files_overwrite_to_partitioned_table_fails_with_lower_and_upper_mismatch(
    spark: SparkSession, session_catalog: Catalog, format_version: int
) -> None:
    identifier = f"default.partitioned_table_mismatch_fails_v{format_version}"

    partition_spec = PartitionSpec(
        PartitionField(source_id=4, field_id=1000, transform=IdentityTransform(), name="baz"),
        spec_id=0,
    )

    tbl = _create_table(session_catalog, identifier, format_version, partition_spec)

    file_paths = [f"s3://warehouse/default/partitioned_table_mismatch_fails/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "foo": True,
                                "bar": "bar_string",
                                "baz": 123,
                                "qux": date(2024, 3, 7),
                            },
                            {
                                "foo": True,
                                "bar": "bar_string",
                                "baz": 124,
                                "qux": date(2024, 3, 7),
                            },
                        ],
                        schema=ARROW_SCHEMA,
                    )
                )

    # add the parquet files as data files
    with pytest.raises(ValueError) as exc_info:
        tbl.add_files_overwrite(file_paths=file_paths)
    assert (
        "Cannot infer partition value from parquet metadata as there are more than one partition values for Partition Field: baz. lower_value=123, upper_value=124"
        in str(exc_info.value)
    )


@pytest.mark.integration
def test_add_files_overwrite_snapshot_properties(spark: SparkSession, session_catalog: Catalog, format_version: int) -> None:
    identifier = f"default.unpartitioned_table_v{format_version}"
    tbl = _create_table(session_catalog, identifier, format_version)

    file_paths = [f"s3://warehouse/default/unpartitioned/v{format_version}/test-{i}.parquet" for i in range(5)]
    # write parquet files
    for file_path in file_paths:
        fo = tbl.io.new_output(file_path)
        with fo.create(overwrite=True) as fos:
            with pq.ParquetWriter(fos, schema=ARROW_SCHEMA) as writer:
                writer.write_table(ARROW_TABLE)

    # add the parquet files as data files
    tbl.add_files_overwrite(file_paths=file_paths, snapshot_properties={"snapshot_prop_a": "test_prop_a"})

    # NameMapping must have been set to enable reads
    assert tbl.name_mapping() is not None

    summary = spark.sql(f"SELECT * FROM {identifier}.snapshots;").collect()[0].summary

    assert "snapshot_prop_a" in summary
    assert summary["snapshot_prop_a"] == "test_prop_a"
