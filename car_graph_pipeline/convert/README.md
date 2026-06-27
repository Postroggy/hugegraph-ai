# Convert

This module converts car-graph raw extraction results into HugeGraph-friendly
vertex and edge files.

## Why This Exists

The extraction workflows output raw graph facts:

```json
{
  "entities": [],
  "relations": []
}
```

HugeGraph import expects vertices and edges with labels, ids, endpoints, and
properties. Conversion is kept as a separate adapter so extraction quality logic
does not mix with graph storage serialization.

## Files

- `to_hugegraph_v2.py`: vehicle-scoped converter for the newer schema where
  labels and relations carry `vehicle_model`.
- `to_hugegraph.py`: older converter without vehicle scope, kept for reference.

## Notes

The copied converters still contain paths and assumptions from the original
`/Users/lzj/proj/car_graph/car_graph_pipeline` workspace. Before using this
module as a production CLI under HugeGraph AI, parameterize input/output paths
and align the converter with the final chosen vehicle-scope schema.

