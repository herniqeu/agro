"use client";
import React from "react";
import { useFormContext } from "react-hook-form";
import { FormLabel, FormItem } from "@/components/ui/form";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";

type ImageUploadFieldProps = {
  handleFileChange: (file: File | null) => void;
};

export const ImageUploadField: React.FC<ImageUploadFieldProps> = ({
  handleFileChange,
}) => {
  const { register } = useFormContext();

  return (
    <FormItem>
      <FormLabel htmlFor="file" className="flex gap-1">
        Image <span className="text-red-500">*</span>
        <Tooltip>
          <TooltipTrigger asChild>
            <HelpCircle color="#71717A" size={16} />
          </TooltipTrigger>
          <TooltipContent>
            <div className="rounded-md p-4 bg-white shadow-sm border-[#e4e4e7] border-[1px]">
              <p>Image for plot detection</p>
            </div>
          </TooltipContent>
        </Tooltip>
      </FormLabel>
      <input
        {...register("file")}
        type="file"
        id="file"
        required
        accept="image/*"
        onChange={(e) => {
          handleFileChange(e.target.files?.[0] || null);
        }}
        className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-slate-100 file:text-slate-700 hover:file:bg-slate-200"
      />
    </FormItem>
  );
};
