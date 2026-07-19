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

"""Abstract LLM errors for the benchmark LLM-Judge layer.

Provider-specific exceptions (e.g. openai SDK errors) are translated into
these at the client boundary (``JudgeLLM.generate`` in ``client.py``) so the
retry logic (``retry_llm_call``) and metrics depend only on this abstract
contract, never on any SDK. A future provider swap only touches ``client.py``.
"""


class LLMError(Exception):
    """Base class for LLM-Judge call failures."""


class LLMTransientError(LLMError):
    """A transient failure worth retrying (rate limit, timeout, 5xx, network)."""


class LLMPermanentError(LLMError):
    """A permanent failure — retrying won't help (auth, bad request, 4xx)."""
