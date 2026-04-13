{
  description = "rust environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    devshell.url = "github:numtide/devshell";
  };
  outputs = { nixpkgs, fenix, devshell, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      pkgsFor = system: import nixpkgs { inherit system; overlays = [ fenix.overlays.default devshell.overlays.default ]; };

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        {
          default = pkgs.devshell.mkShell {
            packages = with pkgs; [
              (fenix.packages.${system}.stable.withComponents [
                "cargo"
                "clippy"
                "rust-src"
                "rustc"
                "rustfmt"
                "rust-analyzer"
              ])
              stdenv
              fish
              python3
              uv
            ];
            commands = [
              {
                name = "claude-local";
                command = ''
                  ANTHROPIC_BASE_URL=http://localhost:8000 \
                  CLAUDE_CODE_ATTRIBUTION_HEADER="0" \
                  ANTHROPIC_DEFAULT_OPUS_MODEL=Qwopus3.5-9B-6bit \
                  ANTHROPIC_DEFAULT_SONNET_MODEL=Qwopus3.5-9B-6bit \
                  ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwopus3.5-9B-6bit \
                  claude --model Qwopus3.5-9B-6bit
                '';
              }
              {
                name = "claude-local-qwen3";
                command = ''
                  ANTHROPIC_BASE_URL=http://localhost:8000 \
                  CLAUDE_CODE_ATTRIBUTION_HEADER="0" \
                  ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.5-9B-oQ4 \
                  ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.5-9B-oQ4 \
                  ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.5-9B-oQ4 \
                  claude --model qwen3.5-9B-oQ4
                '';
              }
              {
                name = "claude-local-qwen3-0.8B";
                command = ''
                  ANTHROPIC_BASE_URL=http://localhost:8000 \
                  CLAUDE_CODE_ATTRIBUTION_HEADER="0" \
                  ANTHROPIC_DEFAULT_OPUS_MODEL=Qwopus3.5-9B-6bit \
                  ANTHROPIC_DEFAULT_SONNET_MODEL=Qwen3.5-0.8B-OptiQ-4bit \
                  ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.5-0.8B-OptiQ-4bit \
                  claude --model Qwen3.5-0.8B-OptiQ-4bit
                '';
              }
            ];
          };
        }
      );


    };

}
