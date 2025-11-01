import openai


def main():
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="my-api-key",
    )
    embeddings = client.embeddings.create(
        model="nomic-embed-text",
        input=["hello", "I'm software developer."],
    )
    print(len(embeddings.data[0].embedding))


if __name__ == "__main__":
    main()
