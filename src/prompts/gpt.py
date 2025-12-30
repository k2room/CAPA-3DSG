from pathlib import Path

class GPTprompt():
    def __init__(self, config=None, dataset_name="CAPA"):
        if config is None:
            print("No config provided to GPT prompt")
            raise ValueError("Config is required for GPTprompt initialization")

        elif dataset_name == "CAPA":
            print("Generate CA3DSG")
            md_path_local = config.project_root + "/src/prompts/CA3DSG_local.md"
            md_path_remote = config.project_root + "/src/prompts/CA3DSG_remote.md"
            md_path_spatial = config.project_root + "/src/prompts/CA3DSG_spatial.md"
            md_path_affordance = config.project_root + "/src/prompts/CA3DSG_affordance.md"
            if Path(md_path_local).exists() is False:
                raise FileNotFoundError(f"CA3DSG_local.md not found in {md_path_local}")
            if Path(md_path_remote).exists() is False:
                raise FileNotFoundError(f"CA3DSG_remote.md not found in {md_path_remote}")
            if Path(md_path_spatial).exists() is False:
                raise FileNotFoundError(f"CA3DSG_spatial.md not found in {md_path_spatial}")
            if Path(md_path_affordance).exists() is False:
                raise FileNotFoundError(f"CA3DSG_affordance.md not found in {md_path_affordance}")
            self.system_prompt_local = Path(md_path_local).read_text() 
            self.system_prompt_remote = Path(md_path_remote).read_text() 
            self.system_prompt_spatial = Path(md_path_spatial).read_text() 
            self.system_prompt_affordance = Path(md_path_affordance).read_text()

        else:
            print("Generate Functional 3D Scene Graphs")
            md_path_local = config.project_root + "/src/prompts/Func3DSG_local.md"
            md_path_remote = config.project_root + "/src/prompts/Func3DSG_remote.md"
            if Path(md_path_local).exists() is False:
                raise FileNotFoundError(f"Func3DSG_local.md not found in {md_path_local}")
            if Path(md_path_remote).exists() is False:
                raise FileNotFoundError(f"Func3DSG_remote.md not found in {md_path_remote}")
            self.system_func_local = Path(md_path_local).read_text() 
            self.system_func_remote = Path(md_path_remote).read_text() 

