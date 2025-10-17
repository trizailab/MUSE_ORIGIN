
import os
import re
import json
from pathlib import Path

from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.controller.service import Controller
from browser_use.filesystem.file_system import FileSystem
from browser_use.controller.views import ClickElementAction
from browser_use.controller.registry.views import ActionModel
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.config import get_default_profile, load_browser_use_config, get_default_llm, FlatEnvConfig


class BrowserUse:
    def __init__(self):
        self.config = load_browser_use_config()
        self.browser_session: BrowserSession | None = None
        self.controller: Controller | None = None
        self.file_system: FileSystem | None = None
        self.llm: ChatOpenAI | None = None

    # TODO: Need to expose more path parameters to initialization
    async def _init_browser_session(self, **kwargs):
        """Initialize browser session using config"""
        if self.browser_session:
            return

        # Get profile config
        # profile_config = get_default_profile(self.config)

        # Merge profile config with defaults and overrides
        profile_data = {
            'downloads_path': '/workspace/downloads',
            'wait_between_actions': 0.5,
            'keep_alive': True,
            'user_data_dir': '~/.config/browseruse/profiles/default',
            'is_mobile': False,
            'device_scale_factor': 1.0,
            'disable_security': False,
            # 'headless': True,
            "default": True,
            "allowed_domains": None,
            # **profile_config,  # Config values override defaults
        }

        # Merge any additional kwargs that are valid BrowserProfile fields
        for key, value in kwargs.items():
            profile_data[key] = value

        # Create browser profile
        profile = BrowserProfile(**profile_data)

        # Create browser session
        self.browser_session = BrowserSession(browser_profile=profile)
        await self.browser_session.start()

        # Create controller for direct actions
        self.controller = Controller()

        self.llm = ChatOpenAI(
            model='gemini-2.5-pro',
            api_key=os.getenv('API_KEY'),
            base_url=os.getenv('BASE_URL'),
            temperature=0.7,
            # max_tokens=llm_config.get('max_tokens'),
        )

        # Initialize FileSystem for extraction actions
        file_system_path = profile_data.get('file_system_path', '/workspace/browser-use')
        self.file_system = FileSystem(base_dir=Path(file_system_path).expanduser())

    async def get_axtree(self):
        page = await self.browser_session.get_current_page()
        cdp_session = await page.context.new_cdp_session(page)

        await cdp_session.send('Accessibility.enable')
        await cdp_session.send('DOM.enable')

        ax_tree = await cdp_session.send('Accessibility.getFullAXTree')
        return flatten_axtree_to_str(ax_tree)

    async def get_browser_state(self) -> str:
        """Get current browser state."""
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=False)
        result = {
            'url': state.url,
            'title': state.title,
            'tabs': [{'url': tab.url, 'title': tab.title} for tab in state.tabs],
            'interactive_elements': [],
        }

        # Add interactive elements with their indices
        for index, element in state.selector_map.items():
            raw_str = element.clickable_elements_to_string().replace('\t', '').replace('\n[', '[')
            all_indices = [int(x) for x in re.findall(r'\[(\d+)\]', raw_str)]
            # main_idx = all_indices[0] if all_indices else None
            sub_element_indices = all_indices[1:] if len(all_indices) > 1 else []

            main_tag_match = re.search(r'\[\d+\]<(.*?)\/>', raw_str)
            main_tag_text = '<' + main_tag_match.group(1).strip() + '/>' if main_tag_match else ""

            elem_info = {
                'index': index,
                'tag': element.tag_name,
                'text': main_tag_text,
                # 'sub_element_index': sub_element_indices,
            }
            if element.attributes.get('placeholder'):
                elem_info['placeholder'] = element.attributes['placeholder']
            if element.attributes.get('href'):
                elem_info['href'] = element.attributes['href']
            result['interactive_elements'].append(elem_info)
        return json.dumps(result["interactive_elements"])

    async def extract_content_by_vision(self, query: str) -> str:
        state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=False)
        response = await self.llm.get_client().chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{state.screenshot}"}
                    }
                ]}
            ]
        )
        await self.browser_session.remove_highlights()
        return response.choices[0].message.content

    async def go_to_url(self, url: str, new_tab: bool):
        if not self.browser_session:
            await self._init_browser_session()

        action = self.controller.registry.create_action_model()(go_to_url={"url": url, "new_tab": new_tab})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def wait(self, seconds: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(wait={"seconds": seconds})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def click_element_by_index(self, index: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(click_element_by_index={"index": index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def input_text(self, index: int, text: str):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(input_text={"index": index, "text": text})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def send_keys(self, keys: str):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(send_keys={"keys": keys})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def upload_file(self, index: int, path: str):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(upload_file={"index": index, "path": path})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def go_back(self):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(go_back={})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def scroll(self, down: bool, num_pages: float, index: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(scroll={"down": down, "num_pages": num_pages, "index": index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def list_tabs(self):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=False)
        tabs = [{'index': tab.page_id, 'url': tab.url, 'title': tab.title} for tab in state.tabs]

        await self.browser_session.remove_highlights()
        return tabs

    async def switch_tab(self, tab_index: int,):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(switch_tab={"page_id": tab_index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def close_tab(self, tab_index: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(close_tab={"page_id": tab_index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def get_dropdown_options(self, index: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(get_dropdown_options={"index": index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def select_dropdown_option(self, index: int, text: str):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(select_dropdown_option={"index": index, "text": text})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

# ========================================================================= #
#  UTILS
# ========================================================================= #
def flatten_axtree_to_str(axtree, ignored_roles=None, depth=0, node_idx=0):
    if ignored_roles is None:
        ignored_roles = {"none"}

    nodes = axtree["nodes"]
    node = nodes[node_idx]

    role = node.get("role", {}).get("value", "")
    name = node.get("name", {}).get("value", "")

    if node.get("ignored", False) or role in ignored_roles:
        children_str = ""
        for child_id in node.get("childIds", []):
            child_idx = next((i for i, n in enumerate(nodes) if n["nodeId"] == child_id), None)
            if child_idx is not None:
                children_str += flatten_axtree_to_str(axtree, ignored_roles, depth, child_idx)
        return children_str

    indent = "    " * depth

    props = []
    for prop in node.get("properties", []):
        p_name = prop.get("name", "")
        p_val_dict = prop.get("value")
        if p_val_dict is not None:
            if isinstance(p_val_dict, dict) and "value" in p_val_dict:
                p_value = p_val_dict["value"]
            else:
                p_value = p_val_dict
        else:
            p_value = None
        props.append(f"{p_name}={repr(p_value)}")
    prop_str = (", " + ", ".join(props)) if props else ""

    s = f"{indent}{role} {repr(name)}{prop_str}\n"

    for child_id in node.get("childIds", []):
        child_idx = next((i for i, n in enumerate(nodes) if n["nodeId"] == child_id), None)
        if child_idx is not None:
            s += flatten_axtree_to_str(axtree, ignored_roles, depth+1, child_idx)
    return s
