from pathlib import Path

class GPTprompt():
    def __init__(self, config=None):
        if config is None:
            print("No config provided to GPT prompt")
            self.system_prompt_func = Path("prompts/Func3DSG.md").read_text() 
            self.system_prompt_spat = Path("prompts/Spat3DSG.md").read_text()
        else:
            md_path_local = config.project_root + "/src/prompts/Func3DSG_local.md"
            md_path_remote = config.project_root + "/src/prompts/Func3DSG_remote.md"
            if Path(md_path_local).exists() is False:
                raise FileNotFoundError(f"Func3DSG_local.md not found in {md_path_local}")
            if Path(md_path_remote).exists() is False:
                raise FileNotFoundError(f"Func3DSG_remote.md not found in {md_path_remote}")
            self.system_func_local = Path(md_path_local).read_text() 
            self.system_func_remote = Path(md_path_remote).read_text() 
            # self.system_prompt_spat = Path(md_path).read_text()
