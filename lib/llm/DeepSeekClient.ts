import {
  LLMClient,
  LLMResponse,
  CreateChatCompletionOptions,
  ChatCompletionOptions,
} from "./LLMClient";
import OpenAI, { ClientOptions } from "openai";
import {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
} from "openai/resources/chat";
import { validateZodSchema } from "../utils";
import { AvailableModel } from "@/types/model";
import { LLMCache } from "../cache/LLMCache";

export class DeepSeekClient extends LLMClient {
  public type = "deepseek" as const;
  private client: OpenAI;
  private cache: LLMCache | undefined;
  private enableCaching: boolean;
  public clientOptions: ClientOptions;

  constructor({
    apiKey,
    enableCaching = false,
    cache,
    modelName = "deepseek-chat",
    userProvidedInstructions,
  }: {
    apiKey: string;
    enableCaching?: boolean;
    cache?: LLMCache;
    modelName?: AvailableModel;
    userProvidedInstructions?: string;
  }) {
    super(modelName, userProvidedInstructions);
    this.clientOptions = { baseURL: "https://api.deepseek.com", apiKey };
    this.client = new OpenAI(this.clientOptions);
    this.cache = cache;
    this.enableCaching = enableCaching;
  }

  async createChatCompletion<T = LLMResponse>({
    options: optionsInitial,
    logger,
    retries = 3,
  }: CreateChatCompletionOptions): Promise<T> {
    const options: Partial<ChatCompletionOptions> = optionsInitial;
    const { requestId, ...optionsWithoutImageAndRequestId } = options;

    const cacheOptions = {
      model: this.modelName,
      messages: options.messages,
      temperature: options.temperature,
      top_p: options.top_p,
      frequency_penalty: options.frequency_penalty,
      presence_penalty: options.presence_penalty,
      response_model: options.response_model,
    };

    if (this.enableCaching) {
      const cachedResponse = await this.cache?.get<T>(cacheOptions, requestId);
      if (cachedResponse) {
        logger({
          category: "llm_cache",
          message: "LLM cache hit - returning cached response",
          level: 1,
          auxiliary: {
            requestId: { value: requestId, type: "string" },
            cachedResponse: {
              value: JSON.stringify(cachedResponse),
              type: "object",
            },
          },
        });
        return cachedResponse;
      }
    }

    logger({
      category: "deepseek",
      message: "creating chat completion",
      level: 1,
      auxiliary: {
        options: {
          value: JSON.stringify({
            ...optionsWithoutImageAndRequestId,
            requestId,
          }),
          type: "object",
        },
      },
    });

    if (options.response_model) {
      options.messages.unshift({
        role: "system",
        content: `Return response in this JSON format: ${JSON.stringify(
          options.response_model.schema,
        )}. Do not include any other text or markdown formatting.`,
      });
    }

    const formattedMessages: ChatCompletionMessageParam[] =
      options.messages.map((message) => ({
        role: message.role,
        content: Array.isArray(message.content)
          ? message.content
              .map((c) => ("text" in c ? c.text : ""))
              .filter(Boolean)
              .join("\n")
          : message.content,
      }));

    const body: ChatCompletionCreateParamsNonStreaming = {
      model: this.modelName,
      messages: formattedMessages,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      top_p: options.top_p,
      frequency_penalty: options.frequency_penalty,
      presence_penalty: options.presence_penalty,
      stream: false,
      tools: options.tools?.map((tool) => ({
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
        type: "function",
      })),
    };

    try {
      const response = await this.client.chat.completions.create(body);

      logger({
        category: "deepseek",
        message: "response received",
        level: 1,
        auxiliary: {
          response: { value: JSON.stringify(response), type: "object" },
          requestId: { value: requestId, type: "string" },
        },
      });

      if (options.response_model) {
        const extractedData = response.choices[0].message.content;
        if (!extractedData) {
          throw new Error("No content in response");
        }
        const parsedData = JSON.parse(extractedData);

        if (!validateZodSchema(options.response_model.schema, parsedData)) {
          if (retries > 0) {
            return this.createChatCompletion({
              options: optionsInitial,
              logger,
              retries: retries - 1,
            });
          }
          throw new Error("Invalid response schema");
        }

        if (this.enableCaching) {
          this.cache?.set(cacheOptions, parsedData, requestId);
        }

        return parsedData;
      }

      if (this.enableCaching) {
        this.cache?.set(cacheOptions, response, requestId);
      }

      return response as T;
    } catch (error) {
      logger({
        category: "deepseek",
        message: "error creating chat completion",
        level: 0,
        auxiliary: {
          error: { value: error.message || "Unknown error", type: "string" },
          code: { value: error.code || "unknown", type: "string" },
          details: {
            value: JSON.stringify(error.error || {}),
            type: "object",
          },
        },
      });

      if (error.status === 402) {
        throw new Error(
          "DeepSeek API: Insufficient balance. Please add funds to your account.",
        );
      }

      if (retries > 0) {
        return this.createChatCompletion({
          options: optionsInitial,
          logger,
          retries: retries - 1,
        });
      }

      throw error;
    }
  }
}
