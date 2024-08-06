import { z } from "zod";

export const formSchema = z.object({
  model: z.string().nonempty({ message: "Please select a model." }),
  file: z.instanceof(File, { message: "Please upload a file." }).optional(),
});

export type FormData = z.infer<typeof formSchema>;
