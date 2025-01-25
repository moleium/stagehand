import { Stagehand } from "../lib";
import { DeepSeekClient } from "../lib/llm/DeepSeekClient";
import { z } from "zod";

async function example() {
  const stagehand = new Stagehand({
    env: "LOCAL",
    verbose: 1,
    debugDom: true,
    enableCaching: false,
    llmClient: new DeepSeekClient({
      apiKey: process.env.DEEPSEEK_API_KEY!,
      modelName: "deepseek-chat",
    }),
  });

  await stagehand.init();
  await stagehand.page.goto("https://news.ycombinator.com");

  const headlines = await stagehand.page.extract({
    instruction: "Extract only 3 stories from the Hacker News homepage.",
    schema: z.object({
      stories: z
        .array(
          z.object({
            title: z.string(),
            url: z.string(),
            points: z.number(),
          }),
        )
        .length(3),
    }),
  });

  console.log(headlines);
  await stagehand.close();
}

(async () => {
  await example();
})();
