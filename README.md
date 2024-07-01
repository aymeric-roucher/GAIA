# Beating GAIA with Transformers Agents 🚀

This is the exact code used for our submission that scores #2 on the test set, #1 on the validation set.

![GAIA leaderboard screenshot](figures/leaderboard.png)

Check out the current leaderboard [here](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

### How to run tests?

First, install requirements:
```bash
pip install -r requirements.txt
```

Setup your secrets in a `.env`file:
```bash
HUGGINGFACEHUB_API_TOKEN
SERPAPI_API_KEY
OPENAI_API_KEY
ANTHROPIC_API_KEY
```

And optionally if you want to use Anthropic models via AWS bedrock:
```bash
AWS_BEDROCK_ID
AWS_BEDROCK_KEY
```

Then run `gaia.py` to launch tests!