import openai


def main():
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="my-api-key",
    )

    response = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": "あなたは優秀なアシスタントです。"},
            {"role": "user", "content": "こんにちは、あなたについて教えてください。"},
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
