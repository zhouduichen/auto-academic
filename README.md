# auto-academic

Mac control client and shared API contract for AutoResearch Workbench.

The Mac client never runs CUDA training or persistent Workbench services. The pinned
`karpathy/autoresearch` source executes only on the Windows CUDA node.

## Development

```bash
uv sync --dev
make quality
```
