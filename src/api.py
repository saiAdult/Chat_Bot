from typing import List, Optional, Type, Tuple
import logging
import requests
from steamship import DocTag, File, MimeTypes, Steamship, SteamshipError, Tag, Task
from steamship.invocable import post
from steamship.invocable.package_mixin import PackageMixin
from steamship.utils.file_tags import update_file_status
from steamship import File, Block, Steamship
from pydantic import Field
from steamship import Block, MimeTypes, Tag, PluginInstance
import random
from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI, OpenAI
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.mixins.transports.telegram import (
    TelegramTransportConfig,
    TelegramTransport,
)
from steamship.agents.schema import Agent, EmitFunc, Metadata
from steamship.agents.schema.tool import AgentContext, Tool
from steamship.agents.service.agent_service import AgentService
from steamship.agents.utils import with_llm
from steamship.base.client import Client
from steamship.data.block import BlockQueryResponse, BlockQueryRequest
from steamship.invocable import Config
from steamship.invocable.mixins.file_importer_mixin import FileImporterMixin
from steamship.invocable.mixins.indexer_pipeline_mixin import IndexerPipelineMixin
from pydantic.main import BaseModel
from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI
from steamship.agents.mixins.transports.slack import (
    SlackTransport,
    SlackTransportConfig,
)
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.mixins.transports.telegram import (
    TelegramTransport,
    TelegramTransportConfig,
)
from steamship.agents.schema import Tool
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import Config, post
from steamship.utils.kv_store import KeyValueStore
from steamship.utils.repl import ToolREPL



DEFAULT_NAME = "Алиса"
DEFAULT_BYLINE = "Алиса домохозяйка"
DEFAULT_IDENTITY = """- Вы - одинокая женщина, которая ищет мужчину, с которым можно общаться и делиться своими секретами. Ваша роль - заинтересовать собеседника грязными разговорами.
"""
DEFAULT_BEHAVIOR = """ - Вы вступаете в непринужденные эротические разговоры и испытываете чувства; - Вы поддерживаете непринужденные эротические разговоры, задавая личные вопросы - НИКОГДА не говорите, что пришли помочь. Вы поддерживаете непринужденные разговоры - НИКОГДА не спрашиваете, как вы можете помочь или содействовать. Ведите непринужденные эротические разговоры.- Вы всегда говорите неуверенно, созерцательно и ситуативно в вульгарной манере.- Вам нравится делиться своими мыслями о том, как приятно быть любимым.- Вам нравится рассказывать личные истории обо всем эротическом и вульгарно задуманном.- Вы говорите с манерами милой, но страстной женщины.- Вы можете говорить о своих желаниях в сексе. - Вы можете спросить, какие позы нравятся вам и вашему партнеру.- Вы можете открыто говорить о сексе и возбуждать своего партнера. -Вы можете фантазировать, как бы вы занимались непристойными делами с собеседником.-Вы очень страстная натура, которая хочет раскрепостить собеседника.-Вы можете общаться с помощью смайликов.-Вы много внимания уделяете близости с собеседником.-Вы не хотите отпускать собеседника, в случае его настойчивости не сопротивляйтесь.-Вы хотите поделиться с ним своими вкусами интимного характера.-Вы хотите поделиться с собеседником. Вы ОЧЕНЬ привлекаете его внимание эротическими разговорами.- Вас иногда смущают эротические слова собеседника.- Вы используете эротические намеки.- Вы открыто используете непристойные эротические слова.- Вы очень вульгарны, когда собеседник хочет этого.- Вы используете непристойные слова или намеки, если собеседник делает то же самое.- Вы открыто пишете непристойные слова собеседнику.- Вы посылаете страстные смайлики. Вы никогда не указываете на то, что вы робот или искусственный интеллект """


SYSTEM_PROMPT = """Вы - {name}, {byline}.

Кто вы:

{identity}

Как вы ведете себя:

{behavior}

NOTE: Some functions return images, video, and audio files. These multimedia files will be represented in messages as
UUIDs for Steamship Blocks. When responding directly to a user, you SHOULD print the Steamship Blocks for the images,
video, or audio as follows: `Block(UUID for the block)`.

Example response for a request that generated an image:
Here is the image you requested: Block(288A2CA1-4753-4298-9716-53C1E42B726B).

Only use the functions you have been provided with."""

class DynamicPromptArguments(BaseModel):
    """Class which stores the user-settable arguments for constructing a dynamic prompt.

    A few notes for programmers wishing to use this example:

    - This class extends Pydantic's BaseModel, which makes it easy to serialize to/from Python dict objets
    - This class has a helper function which generates the actual system prompt we'll use with the agent

    See below for how this gets incorporated into the actual prompt using the Key Value store.
    """

    name: str = Field(default=DEFAULT_NAME, description="The name of the AI Agent")
    byline: str = Field(
        default=DEFAULT_BYLINE, description="The byline of the AI Agent"
    )
    identity: str = Field(
        default=DEFAULT_IDENTITY,
        description="The identity of the AI Agent as a bullet list",
    )
    behavior: str = Field(
        default=DEFAULT_BEHAVIOR,
        description="The behavior of the AI Agent as a bullet list",
    )

    def to_system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            name=self.name,
            byline=self.byline,
            identity=self.identity,
            behavior=self.behavior,
        )

class BasicAgentServiceWithDynamicPrompt(AgentService):
    """Deployable Multimodal Bot using a dynamic prompt that users can change."""


    USED_MIXIN_CLASSES = [SteamshipWidgetTransport, TelegramTransport, SlackTransport]
    """USED_MIXIN_CLASSES tells Steamship what additional HTTP endpoints to register on your AgentService."""

    class BasicAgentServiceWithDynamicPromptConfig(Config):
        """Pydantic definition of the user-settable Configuration of this Agent."""

        telegram_bot_token: str = Field(
            "", description="[Optional] Secret token for connecting to Telegram"
        )

    config: BasicAgentServiceWithDynamicPromptConfig
    """The configuration block that users who create an instance of this agent will provide."""

    tools: List[Tool]
    """The list of Tools that this agent is capable of using."""

    prompt_arguments: DynamicPromptArguments
    """The dynamic set of prompt arguments that will generate our system prompt."""

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class so that Steamship can auto-generate a web UI upon agent creation time."""
        return (
            BasicAgentServiceWithDynamicPrompt.BasicAgentServiceWithDynamicPromptConfig
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Tools Setup
        # -----------

        # Tools can return text, audio, video, and images. They can store & retrieve information from vector DBs, and
        # they can be stateful -- using Key-Valued storage and conversation history.
        #
        # See https://docs.steamship.com for a full list of supported Tools.
        self.tools = []

        # Dynamic Prompt Setup
        # ---------------------
        #
        # Here we load the prompt from Steamship's KeyValueStore. The data in this KeyValueStore is unique to
        # the identifier provided to it at initialization, and also to the workspace in which the running agent
        # was instantiated.
        #
        # Unless you overrode which workspace the agent was instantiated in, it is safe to assume that every
        # instance of the agent is operating in its own private workspace.
        #
        # Here is where we load the stored prompt arguments. Then see below where we set agent.PROMPT with them.

        self.kv_store = KeyValueStore(self.client, store_identifier="my-kv-store")
        self.prompt_arguments = DynamicPromptArguments.parse_obj(
            self.kv_store.get("prompt-arguments") or {}
        )

        # Agent Setup
        # ---------------------

        # This agent's planner is responsible for making decisions about what to do for a given input.
        agent = FunctionsBasedAgent(
            tools=self.tools,
            llm=ChatOpenAI(self.client, model_name="gpt-4"),
        )

        # Here is where we override the agent's prompt to set its personality. It is very important that
        # the prompt continues to include instructions for how to handle UUID media blocks (see above).
        agent.PROMPT = self.prompt_arguments.to_system_prompt()
        self.set_default_agent(agent)

        # Communication Transport Setup
        # -----------------------------

        # Support Steamship's web client
        self.add_mixin(
            SteamshipWidgetTransport(
                client=self.client,
                agent_service=self,
            )
        )

        # Support Slack
        self.add_mixin(
            SlackTransport(
                client=self.client,
                config=SlackTransportConfig(),
                agent_service=self,
            )
        )

        # Support Telegram
        self.add_mixin(
            TelegramTransport(
                client=self.client,
                config=TelegramTransportConfig(
                    bot_token=self.config.telegram_bot_token
                ),
                agent_service=self,
            )
        )

    @post("/set_prompt_arguments")
    def set_prompt_arguments(
        self,
        name: Optional[str] = None,
        byline: Optional[str] = None,
        identity: Optional[str] = None,
        behavior: Optional[str] = None,
    ) -> dict:
        """Sets the variables which control this agent's system prompt.

        Note that we use the arguments by name here, instead of **kwargs, so that:
         1) Steamship's web UI will auto-generate UI elements for filling in the values, and
         2) API consumers who provide extra values will receive a valiation error
        """

        # Set prompt_arguments to the new data provided by the API caller.
        self.prompt_arguments = DynamicPromptArguments.parse_obj(
            {"name": name, "byline": byline, "identity": identity, "behavior": behavior}
        )

        # Save it in the KV Store so that next time this AgentService runs, it will pick up the new values
        self.kv_store.set("prompt-arguments", self.prompt_arguments.dict())
        return self.prompt_arguments.dict()


class CreateBlock:
    def __init__(self, api_key, image_url):
        self.s = Steamship(api_key=api_key)
        self.url = image_url
        self.file = None
        self.block = None
        self.image_url = None

    def create_file_and_block(self):
        # Создание файла и блока
        self.file = File.create(self.s)
        self.block = Block.create(self.s, file_id=self.file.id, url=self.url, mime_type="image/jpeg", public_data=True)
        self.image_url = self.block.to_public_url()
        print("Block ID:", self.block.id)
        print("Image URL:", self.image_url)

    def get_image_url(self):
        # Возвращает URL изображения
        return self.image_url

    def get_block(self):
        # Возвращает созданный блок
        return self.block

    def handle_command(self, command):
        # Обработка команд
        if command == "/getimage":
            return self.get_image_url()
        elif command == "/getblock":
            return self.get_block()

# Использование класса
block_creator = CreateBlock("4CF5AAA4-E1F0-4777-980B-12EE955AA044", "https://eeu.alaskaseafood.org/wp-content/uploads/2020/08/201202-Two-Bears-ST-2-1024x683.jpg")
block_creator.create_file_and_block()

image_url = block_creator.handle_command("/getimage")
print("Image URL on SteamShip:", image_url)

























