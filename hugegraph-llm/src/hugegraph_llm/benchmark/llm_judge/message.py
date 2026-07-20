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

"""Message type for chat-completions calls.

``JudgeLLM.generate`` takes a list of ``Message`` rather than a freeform prompt
string. The role defaults to "user" — current LLM-Judge metrics send a single
user prompt without restructuring it into system/assistant turns, so callers
just write ``Message(prompt)``. The full role set is supported on the type so
system/assistant turns can be added later without changing the signature.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Message:
    """A single chat-completions message (role defaults to "user")."""

    content: str
    role: Literal["system", "user", "assistant"] = "user"

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}
