"""Fail closed when fixture transport setup did not install its in-memory substitute."""

raise ImportError("The offline fixture input was not installed. Physical PortAudio is disabled in this test.")
