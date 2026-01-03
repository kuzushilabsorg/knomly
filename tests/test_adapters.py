"""
Tests for the Adapters Layer.

Tests cover:
- ToolDefinition schema and serialization
- ProviderDefinition schema
- PipelinePacket schema
- ServiceRegistry and GenericServiceFactory
- ToolBuilder with adapters
"""

import pytest

from knomly.adapters.base import (
    BaseToolAdapter,
    DictServiceRegistry,
    ToolBuilder,
)
from knomly.adapters.openapi_adapter import (
    OpenAPISpecImporter,
    OpenAPIToolAdapter,
    SpecCache,
)
from knomly.adapters.schemas import (
    PipelinePacket,
    ProviderDefinition,
    ToolDefinition,
    ToolParameter,
)
from knomly.adapters.service_factory import (
    GenericServiceFactory,
)

# =============================================================================
# Test ToolDefinition Schema
# =============================================================================


class TestToolDefinition:
    """Tests for ToolDefinition schema."""

    def test_basic_creation(self):
        """Test basic tool definition creation."""
        tool = ToolDefinition(
            name="create_task",
            description="Create a new task",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                },
                "required": ["title"],
            },
        )

        assert tool.name == "create_task"
        assert tool.description == "Create a new task"
        assert tool.source == "native"  # default
        assert tool.enabled is True  # default

    def test_json_serialization(self):
        """Test tool can be serialized to/from JSON."""
        tool = ToolDefinition(
            name="search_tasks",
            description="Search for tasks",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            source="openapi",
            source_config={
                "spec_url": "https://api.example.com/openapi.json",
                "operation_id": "searchTasks",
            },
            tags=["tasks", "search"],
        )

        # Serialize to JSON
        json_str = tool.model_dump_json()
        assert "search_tasks" in json_str
        assert "openapi" in json_str

        # Deserialize from JSON
        restored = ToolDefinition.model_validate_json(json_str)
        assert restored.name == tool.name
        assert restored.source == "openapi"
        assert restored.source_config["operation_id"] == "searchTasks"

    def test_structured_params_to_schema(self):
        """Test conversion of structured params to JSON Schema."""
        tool = ToolDefinition(
            name="create_task",
            description="Create a task",
            structured_params=[
                ToolParameter(
                    name="title",
                    type="string",
                    description="Task title",
                    required=True,
                ),
                ToolParameter(
                    name="priority",
                    type="string",
                    description="Priority level",
                    required=False,
                    enum=["low", "medium", "high"],
                ),
            ],
        )

        schema = tool.get_parameters_schema()
        assert schema["type"] == "object"
        assert "title" in schema["properties"]
        assert "priority" in schema["properties"]
        assert "title" in schema["required"]
        assert "priority" not in schema["required"]
        assert schema["properties"]["priority"]["enum"] == ["low", "medium", "high"]

    def test_to_openai_tool(self):
        """Test conversion to OpenAI tool format."""
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )

        openai_tool = tool.to_openai_tool()
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "get_weather"
        assert "location" in openai_tool["function"]["parameters"]["properties"]

    def test_from_function_schema(self):
        """Test creation from Pipecat-style function schema."""
        tool = ToolDefinition.from_function_schema(
            name="send_email",
            description="Send an email",
            properties={
                "to": {"type": "string", "description": "Recipient email"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            required=["to", "subject", "body"],
            source="native",
        )

        assert tool.name == "send_email"
        assert tool.source == "native"
        assert "to" in tool.parameters["properties"]

    def test_from_openapi_operation(self):
        """Test creation from OpenAPI operation."""
        tool = ToolDefinition.from_openapi_operation(
            operation_id="createWorkItem",
            method="post",
            path="/api/v1/projects/{project_id}/tasks",
            summary="Create a new task in a project",
            parameters=[
                {
                    "name": "project_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Project ID",
                },
            ],
            request_body_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title"],
            },
            tags=["tasks"],
            spec_url="https://api.plane.so/openapi.json",
            base_url="https://api.plane.so",
        )

        assert tool.name == "createWorkItem"
        assert tool.source == "openapi"
        assert "project_id" in tool.parameters["properties"]
        assert "title" in tool.parameters["properties"]
        assert tool.destructive is False
        assert tool.read_only is False


class TestToolParameter:
    """Tests for ToolParameter."""

    def test_to_json_schema(self):
        """Test conversion to JSON Schema property."""
        param = ToolParameter(
            name="priority",
            type="string",
            description="Task priority",
            enum=["low", "medium", "high"],
            default="medium",
        )

        schema = param.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "Task priority"
        assert schema["enum"] == ["low", "medium", "high"]
        assert schema["default"] == "medium"


# =============================================================================
# Test ProviderDefinition Schema
# =============================================================================


class TestProviderDefinition:
    """Tests for ProviderDefinition schema."""

    def test_stt_factory(self):
        """Test STT provider factory."""
        provider = ProviderDefinition.stt(
            "deepgram",
            model="nova-2",
            language="en",
            auth_secret_key="deepgram_api_key",
        )

        assert provider.provider_type == "stt"
        assert provider.provider_code == "deepgram"
        assert provider.params["model"] == "nova-2"
        assert provider.auth_secret_key == "deepgram_api_key"

    def test_llm_factory(self):
        """Test LLM provider factory."""
        provider = ProviderDefinition.llm(
            "openai",
            model="gpt-4o",
            temperature=0.5,
            auth_secret_key="openai_api_key",
        )

        assert provider.provider_type == "llm"
        assert provider.provider_code == "openai"
        assert provider.params["model"] == "gpt-4o"
        assert provider.params["temperature"] == 0.5

    def test_get_auth(self):
        """Test getting auth from secrets."""
        provider = ProviderDefinition.stt(
            "deepgram",
            auth_secret_key="dg_key",
        )

        secrets = {"dg_key": "secret123", "other": "value"}
        assert provider.get_auth(secrets) == "secret123"

        # Missing key
        assert provider.get_auth({}) is None

    def test_json_serialization(self):
        """Test JSON serialization."""
        provider = ProviderDefinition(
            provider_type="llm",
            provider_code="anthropic",
            params={"model": "claude-3-opus", "max_tokens": 4096},
            auth_secret_key="anthropic_key",
            priority=10,
        )

        json_str = provider.model_dump_json()
        restored = ProviderDefinition.model_validate_json(json_str)

        assert restored.provider_code == "anthropic"
        assert restored.params["model"] == "claude-3-opus"
        assert restored.priority == 10


# =============================================================================
# Test PipelinePacket Schema
# =============================================================================


class TestPipelinePacket:
    """Tests for PipelinePacket schema."""

    def test_create_for_session(self):
        """Test factory method."""
        packet = PipelinePacket.create_for_session(
            session_id="sess-123",
            user_id="user-456",
            room_name="room-789",
            system_prompt="You are a helpful assistant.",
            welcome_message="Hello!",
        )

        assert packet.session.session_id == "sess-123"
        assert packet.session.user_id == "user-456"
        assert packet.agent.system_prompt == "You are a helpful assistant."
        assert packet.providers.llm is not None
        assert packet.providers.llm.provider_code == "openai"

    def test_with_tools(self):
        """Test packet with tool definitions."""
        tools = [
            ToolDefinition(
                name="create_task",
                description="Create a task",
            ),
            ToolDefinition(
                name="search_tasks",
                description="Search tasks",
                enabled=False,
            ),
        ]

        packet = PipelinePacket.create_for_session(
            session_id="sess-1",
            user_id="user-1",
            tools=tools,
        )

        assert len(packet.tools) == 2
        enabled = packet.get_enabled_tools()
        assert len(enabled) == 1
        assert enabled[0].name == "create_task"

    def test_to_redis_key(self):
        """Test Redis key generation."""
        packet = PipelinePacket.create_for_session(
            session_id="sess-abc",
            user_id="user-123",
        )

        key = packet.to_redis_key()
        assert key == "pipeline:sess-abc"

    def test_json_roundtrip(self):
        """Test full JSON serialization roundtrip."""
        packet = PipelinePacket.create_for_session(
            session_id="sess-test",
            user_id="user-test",
            system_prompt="Test prompt",
            stt=ProviderDefinition.stt("deepgram"),
            llm=ProviderDefinition.llm("openai"),
            tools=[
                ToolDefinition(name="tool1", description="Tool 1"),
            ],
        )

        json_str = packet.model_dump_json()
        restored = PipelinePacket.model_validate_json(json_str)

        assert restored.session.session_id == "sess-test"
        assert restored.agent.system_prompt == "Test prompt"
        assert len(restored.tools) == 1
        assert restored.providers.stt.provider_code == "deepgram"


# =============================================================================
# Test Service Registry
# =============================================================================


class TestDictServiceRegistry:
    """Tests for DictServiceRegistry."""

    def test_get_config(self):
        """Test getting provider config."""

        class MockSTT:
            pass

        registry = DictServiceRegistry(
            {
                "stt": {
                    "deepgram": {
                        "class": MockSTT,
                        "auth": {"arg": "api_key"},
                    },
                },
            }
        )

        config = registry.get_config("stt", "deepgram")
        assert config is not None
        assert config["class"] == MockSTT

        # Unknown provider
        assert registry.get_config("stt", "unknown") is None

    def test_list_providers(self):
        """Test listing providers."""
        registry = DictServiceRegistry(
            {
                "stt": {
                    "deepgram": {},
                    "whisper": {},
                },
                "llm": {
                    "openai": {},
                },
            }
        )

        stt_providers = registry.list_providers("stt")
        assert "deepgram" in stt_providers
        assert "whisper" in stt_providers
        assert len(stt_providers) == 2

    def test_register_unregister(self):
        """Test dynamic registration."""
        registry = DictServiceRegistry({})

        registry.register("stt", "test", {"class": str})
        assert registry.get_config("stt", "test") is not None

        registry.unregister("stt", "test")
        assert registry.get_config("stt", "test") is None


# =============================================================================
# Test Generic Service Factory
# =============================================================================


class TestGenericServiceFactory:
    """Tests for GenericServiceFactory."""

    def test_create_service(self):
        """Test creating a service from definition."""

        class MockSTT:
            def __init__(self, api_key, model=None, language="en"):
                self.api_key = api_key
                self.model = model
                self.language = language

        registry = DictServiceRegistry(
            {
                "stt": {
                    "mock": {
                        "class": MockSTT,
                        "auth": {"arg": "api_key"},
                        "direct_args": ["model", "language"],
                    },
                },
            }
        )

        factory = GenericServiceFactory(registry)

        definition = ProviderDefinition(
            provider_type="stt",
            provider_code="mock",
            params={"model": "nova-2", "language": "en-US"},
            auth_secret_key="mock_api_key",
        )

        service = factory.create_service(
            definition,
            secrets={"mock_api_key": "secret123"},
        )

        assert service is not None
        assert service.api_key == "secret123"
        assert service.model == "nova-2"
        assert service.language == "en-US"

    def test_create_service_with_params_class(self):
        """Test creating service with nested params class."""

        class MockParams:
            def __init__(self, model, temperature=0.7):
                self.model = model
                self.temperature = temperature

        class MockLLM:
            def __init__(self, api_key, params):
                self.api_key = api_key
                self.params = params

        registry = DictServiceRegistry(
            {
                "llm": {
                    "mock": {
                        "class": MockLLM,
                        "auth": {"arg": "api_key"},
                        "params_class": MockParams,
                        "params_arg": "params",
                    },
                },
            }
        )

        factory = GenericServiceFactory(registry)

        definition = ProviderDefinition(
            provider_type="llm",
            provider_code="mock",
            params={"model": "gpt-4", "temperature": 0.5},
            auth_secret_key="llm_key",
        )

        service = factory.create_service(
            definition,
            secrets={"llm_key": "key123"},
        )

        assert service is not None
        assert service.api_key == "key123"
        assert isinstance(service.params, MockParams)
        assert service.params.model == "gpt-4"
        assert service.params.temperature == 0.5

    def test_unknown_provider_returns_none(self):
        """Test that unknown provider returns None."""
        factory = GenericServiceFactory(DictServiceRegistry({}))

        definition = ProviderDefinition(
            provider_type="stt",
            provider_code="unknown",
        )

        service = factory.create_service(definition)
        assert service is None

    def test_list_providers(self):
        """Test listing supported providers."""
        registry = DictServiceRegistry(
            {
                "stt": {"a": {}, "b": {}},
                "llm": {"x": {}},
            }
        )
        factory = GenericServiceFactory(registry)

        assert factory.list_providers("stt") == ["a", "b"]
        assert factory.list_providers("llm") == ["x"]

    def test_supports_provider(self):
        """Test checking provider support."""
        registry = DictServiceRegistry(
            {
                "stt": {"deepgram": {}},
            }
        )
        factory = GenericServiceFactory(registry)

        assert factory.supports_provider("stt", "deepgram") is True
        assert factory.supports_provider("stt", "whisper") is False


# =============================================================================
# Test Tool Builder
# =============================================================================


class TestToolBuilder:
    """Tests for ToolBuilder."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter that handles 'native' source."""

        class MockTool:
            def __init__(self, name):
                self.name = name
                self.description = f"Mock tool: {name}"

        class MockAdapter(BaseToolAdapter):
            source_type = "native"  # Use valid source type

            async def build_tool(self, definition, context):
                return MockTool(definition.name)

        return MockAdapter()

    @pytest.mark.asyncio
    async def test_build_tool(self, mock_adapter):
        """Test building a single tool."""
        from knomly.tools.factory import ToolContext

        builder = ToolBuilder(adapters={"native": mock_adapter})

        definition = ToolDefinition(
            name="test_tool",
            description="Test",
            source="native",
        )

        context = ToolContext(user_id="user-1")
        tool = await builder.build_tool(definition, context)

        assert tool is not None
        assert tool.name == "test_tool"

    @pytest.mark.asyncio
    async def test_build_tools_filters_disabled(self, mock_adapter):
        """Test that disabled tools are skipped."""
        from knomly.tools.factory import ToolContext

        builder = ToolBuilder(adapters={"native": mock_adapter})

        definitions = [
            ToolDefinition(name="enabled", description="Enabled", source="native"),
            ToolDefinition(name="disabled", description="Disabled", source="native", enabled=False),
        ]

        context = ToolContext(user_id="user-1")
        tools = await builder.build_tools(definitions, context)

        assert len(tools) == 1
        assert tools[0].name == "enabled"

    @pytest.mark.asyncio
    async def test_build_tools_filters_unsupported_source(self, mock_adapter):
        """Test that unsupported sources are skipped."""
        from knomly.tools.factory import ToolContext

        builder = ToolBuilder(adapters={"native": mock_adapter})

        definitions = [
            ToolDefinition(name="supported", description="Supported", source="native"),
            ToolDefinition(name="unsupported", description="Unsupported", source="openapi"),
        ]

        context = ToolContext(user_id="user-1")
        tools = await builder.build_tools(definitions, context)

        assert len(tools) == 1
        assert tools[0].name == "supported"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the adapter layer."""

    @pytest.mark.asyncio
    async def test_full_pipeline_packet_workflow(self):
        """Test creating, serializing, and using a pipeline packet."""
        # 1. Create tool definitions (simulating DB load)
        tool_defs = [
            ToolDefinition(
                name="create_task",
                description="Create a task",
                source="native",
                tags=["tasks"],
            ),
            ToolDefinition(
                name="query_projects",
                description="Query projects",
                source="native",
                tags=["projects"],
            ),
        ]

        # 2. Create provider definitions
        stt = ProviderDefinition.stt("deepgram", model="nova-2")
        llm = ProviderDefinition.llm("openai", model="gpt-4o")

        # 3. Create pipeline packet
        packet = PipelinePacket.create_for_session(
            session_id="test-sess",
            user_id="test-user",
            system_prompt="You are a task manager.",
            stt=stt,
            llm=llm,
            tools=tool_defs,
        )

        # 4. Serialize (as if storing in Redis)
        json_str = packet.model_dump_json()
        assert len(json_str) > 0

        # 5. Deserialize (as if loading from Redis)
        loaded = PipelinePacket.model_validate_json(json_str)

        # 6. Verify all data preserved
        assert loaded.session.session_id == "test-sess"
        assert loaded.agent.system_prompt == "You are a task manager."
        assert loaded.providers.stt.provider_code == "deepgram"
        assert len(loaded.tools) == 2

        # 7. Filter tools by tag
        task_tools = loaded.get_tools_by_tag("tasks")
        assert len(task_tools) == 1
        assert task_tools[0].name == "create_task"


# =============================================================================
# Test SpecCache
# =============================================================================


class TestSpecCache:
    """Tests for SpecCache."""

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test basic cache operations."""
        cache = SpecCache(ttl_seconds=3600)

        spec = {"openapi": "3.0.0", "paths": {}}
        await cache.set("https://api.example.com/spec.json", spec)

        result = await cache.get("https://api.example.com/spec.json")
        assert result == spec

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = SpecCache()

        result = await cache.get("https://unknown.com/spec.json")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        """Test cache entries expire."""
        cache = SpecCache(ttl_seconds=0)  # Immediate expiry

        spec = {"openapi": "3.0.0"}
        await cache.set("https://test.com/spec.json", spec)

        # Should be expired immediately
        result = await cache.get("https://test.com/spec.json")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test cache clearing."""
        cache = SpecCache()

        await cache.set("url1", {"a": 1})
        await cache.set("url2", {"b": 2})
        assert cache.size() == 2

        await cache.clear()
        assert cache.size() == 0


# =============================================================================
# Test OpenAPIToolAdapter
# =============================================================================


class TestOpenAPIToolAdapter:
    """Tests for OpenAPIToolAdapter."""

    @pytest.fixture
    def sample_spec(self):
        """Return a minimal OpenAPI spec for testing."""
        return {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "servers": [{"url": "https://api.test.com"}],
            "paths": {
                "/tasks": {
                    "post": {
                        "operationId": "createTask",
                        "summary": "Create a task",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                        },
                                        "required": ["title"],
                                    }
                                }
                            }
                        },
                        "responses": {"201": {"description": "Created"}},
                    },
                    "get": {
                        "operationId": "listTasks",
                        "summary": "List all tasks",
                        "responses": {"200": {"description": "Success"}},
                    },
                },
            },
        }

    @pytest.mark.asyncio
    async def test_build_tool_from_inline_spec(self, sample_spec):
        """Test building tool from inline spec."""
        from knomly.tools.factory import ToolContext

        adapter = OpenAPIToolAdapter()

        definition = ToolDefinition(
            name="create_task",
            description="Create a task",
            source="openapi",
            source_config={
                "spec": sample_spec,
                "operation_id": "createTask",
                "auth_type": "bearer",
                "auth_secret_key": "api_key",
            },
        )

        context = ToolContext(
            user_id="user-1",
            secrets={"api_key": "test-key-123"},
        )

        tool = await adapter.build_tool(definition, context)

        assert tool is not None
        assert tool.name == "createTask"
        assert "Create a task" in tool.description

    @pytest.mark.asyncio
    async def test_build_tool_with_base_url_override(self, sample_spec):
        """Test that base_url in config overrides spec."""
        from knomly.tools.factory import ToolContext

        adapter = OpenAPIToolAdapter()

        definition = ToolDefinition(
            name="list_tasks",
            description="List tasks",
            source="openapi",
            source_config={
                "spec": sample_spec,
                "operation_id": "listTasks",
                "base_url": "https://custom.api.com",  # Override
            },
        )

        context = ToolContext(user_id="user-1")
        tool = await adapter.build_tool(definition, context)

        assert tool is not None
        assert tool._base_url == "https://custom.api.com"

    @pytest.mark.asyncio
    async def test_build_tool_missing_operation_raises(self, sample_spec):
        """Test that missing operation raises ValueError."""
        from knomly.tools.factory import ToolContext

        adapter = OpenAPIToolAdapter()

        definition = ToolDefinition(
            name="nonexistent",
            description="Does not exist",
            source="openapi",
            source_config={
                "spec": sample_spec,
                "operation_id": "nonExistentOperation",
            },
        )

        context = ToolContext(user_id="user-1")

        with pytest.raises(ValueError, match="not found in spec"):
            await adapter.build_tool(definition, context)

    @pytest.mark.asyncio
    async def test_build_tool_no_spec_raises(self):
        """Test that missing spec raises ValueError."""
        from knomly.tools.factory import ToolContext

        adapter = OpenAPIToolAdapter()

        definition = ToolDefinition(
            name="bad_config",
            description="Bad config",
            source="openapi",
            source_config={
                "operation_id": "something",
                # No spec or spec_url
            },
        )

        context = ToolContext(user_id="user-1")

        with pytest.raises(ValueError, match="Either 'spec' or 'spec_url' required"):
            await adapter.build_tool(definition, context)

    @pytest.mark.asyncio
    async def test_auth_types(self, sample_spec):
        """Test different auth types."""
        from knomly.tools.factory import ToolContext

        adapter = OpenAPIToolAdapter()

        # Test bearer
        definition = ToolDefinition(
            name="test",
            description="Test",
            source="openapi",
            source_config={
                "spec": sample_spec,
                "operation_id": "createTask",
                "auth_type": "bearer",
                "auth_secret_key": "key",
            },
        )
        context = ToolContext(user_id="u", secrets={"key": "bearer-token"})

        auth = adapter._build_auth(definition.source_config, context)
        assert auth == {"api_key": "bearer-token"}

        # Test x_api_key
        definition.source_config["auth_type"] = "x_api_key"
        auth = adapter._build_auth(definition.source_config, context)
        assert auth == {"x_api_key": "bearer-token"}

        # Test basic
        definition.source_config["auth_type"] = "basic"
        context = ToolContext(user_id="u", secrets={"key": "user:pass"})
        auth = adapter._build_auth(definition.source_config, context)
        assert auth == {"basic_user": "user", "basic_pass": "pass"}

    @pytest.mark.asyncio
    async def test_spec_cache_used(self, sample_spec):
        """Test that spec cache is used."""
        cache = SpecCache()
        await cache.set("https://cached.com/spec.json", sample_spec)

        adapter = OpenAPIToolAdapter(spec_cache=cache)

        # When spec_url points to cached location
        definition = ToolDefinition(
            name="test",
            description="Test",
            source="openapi",
            source_config={
                "spec_url": "https://cached.com/spec.json",
                "operation_id": "createTask",
            },
        )

        from knomly.tools.factory import ToolContext

        context = ToolContext(user_id="u")

        # Should use cache (no network call)
        tool = await adapter.build_tool(definition, context)
        assert tool is not None


# =============================================================================
# Test OpenAPISpecImporter
# =============================================================================


class TestOpenAPISpecImporter:
    """Tests for OpenAPISpecImporter."""

    @pytest.mark.asyncio
    async def test_import_from_inline_spec(self):
        """Test importing from inline spec."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.test.com"}],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List users",
                        "tags": ["users"],
                        "responses": {"200": {"description": "OK"}},
                    },
                    "post": {
                        "operationId": "createUser",
                        "summary": "Create user",
                        "tags": ["users"],
                        "responses": {"201": {"description": "Created"}},
                    },
                },
                "/items": {
                    "get": {
                        "operationId": "listItems",
                        "summary": "List items",
                        "tags": ["items"],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        importer = OpenAPISpecImporter()

        # Import all operations
        definitions = await importer.import_spec(
            spec=spec,
            auth_secret_key="api_key",
        )

        assert len(definitions) == 3
        names = [d.name for d in definitions]
        assert "listUsers" in names
        assert "createUser" in names
        assert "listItems" in names

    @pytest.mark.asyncio
    async def test_import_with_tag_filter(self):
        """Test importing with tag filter."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.test.com"}],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "tags": ["users"],
                        "responses": {"200": {}},
                    },
                },
                "/items": {
                    "get": {
                        "operationId": "listItems",
                        "tags": ["items"],
                        "responses": {"200": {}},
                    },
                },
            },
        }

        importer = OpenAPISpecImporter()

        # Import only "users" tag
        definitions = await importer.import_spec(
            spec=spec,
            tags_filter=["users"],
        )

        assert len(definitions) == 1
        assert definitions[0].name == "listUsers"

    @pytest.mark.asyncio
    async def test_imported_definition_has_correct_structure(self):
        """Test that imported definitions have correct source_config."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [{"url": "https://api.test.com"}],
            "paths": {
                "/task": {
                    "post": {
                        "operationId": "createTask",
                        "summary": "Create task",
                        "responses": {"201": {}},
                    },
                },
            },
        }

        importer = OpenAPISpecImporter()

        definitions = await importer.import_spec(
            spec=spec,
            base_url="https://custom.url",
            auth_type="x_api_key",
            auth_secret_key="my_key",
        )

        assert len(definitions) == 1
        defn = definitions[0]

        assert defn.source == "openapi"
        assert defn.source_config["base_url"] == "https://custom.url"
        assert defn.source_config["auth_type"] == "x_api_key"
        assert defn.source_config["auth_secret_key"] == "my_key"
        assert defn.source_config["operation_id"] == "createTask"
