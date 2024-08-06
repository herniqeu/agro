import React, { useEffect, useState } from "react";
import axios from "axios";
import { useFormContext } from "react-hook-form";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import { FormLabel, FormItem } from "@/components/ui/form";

import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";

import { HelpCircle } from "lucide-react";

export const ModelSelectField: React.FC = () => {
  const { register } = useFormContext();
  const [models, setModels] = useState(["Loading"]);

  useEffect(() => {
    axios
      .get("https://a28f-204-199-57-10.ngrok-free.app/api/models", {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      })
      .then((response) => setModels(response.data.models));
  }, []);

  return (
    <FormItem>
      <FormLabel htmlFor="model" className="flex gap-1">
        Model <span className="text-red-500">*</span>
        <Tooltip>
          <TooltipTrigger asChild>
            <HelpCircle color="#71717A" size={16} />
          </TooltipTrigger>
          <TooltipContent>
            <div className="rounded-md p-4 bg-white shadow-sm border-[#e4e4e7] border-[1px]">
              <p>The model you want to use</p>
            </div>
          </TooltipContent>
        </Tooltip>
      </FormLabel>
      <Select {...register("model")} required>
        <SelectTrigger className="w-[35%]">
          <SelectValue placeholder="Select a model" />
        </SelectTrigger>
        <SelectContent>
          {models.map((model, index) => (
            <SelectItem key={index} value={model}>
              {model}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </FormItem>
  );
};
