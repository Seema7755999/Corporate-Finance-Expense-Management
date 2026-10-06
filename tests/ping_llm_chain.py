'''Check the API keys in .env: send one short message to every model of the LedgerLens chain.

One working model is enough for the app, so the exit code is non-zero only if ALL of them fail.
Run from the project folder:   python tests/ping_llm_chain.py
'''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):  # never crash on old Windows console encodings
    sys.stdout.reconfigure(errors='replace')

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from llm import active_providers, build_chat_model, describe_error


def main() -> int:
    providers = active_providers()
    if not providers:
        print('No usable API key found. Put GOOGLE_API_KEY and/or GROQ_API_KEY into the .env file.')
        return 1
    failures = 0
    for provider in providers:
        print(f'Testing {provider.label} (model: {provider.model}) ...')
        try:
            reply = build_chat_model(provider, timeout=30).invoke('Say hello in five words.')
            print(f'  [OK] works - reply: {str(reply.text).strip()}')  # .text is a property, not a method
        except Exception as exc:  # noqa: BLE001 - report every model, whatever its failure
            failures += 1
            print(f'  [FAILED] {describe_error(exc)}')
    return 1 if failures == len(providers) else 0


if __name__ == '__main__':
    sys.exit(main())
