"use client";
import axios from "axios";
import React from "react";
import { useFormContext } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { ModelSelectField } from "@/components/model-select-field";
import { ImageUploadField } from "@/components/image-upload-field";

type FormContainerProps = {
  onSubmit: (data: any) => void;
  handleFileChange: (file: File | null) => void;
};

export const FormContainer: React.FC<FormContainerProps> = ({
  handleFileChange,
}) => {
  const { handleSubmit, watch } = useFormContext();
  const model = watch("model");

  const onSubmit = (data: any) => {
    const formData = new FormData();
    formData.append("file", data.file);

    axios
      .post(
        `https://a28f-204-199-57-10.ngrok-free.app/api/models/${model}/predict`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      )
      .then((response) => {
        // handle response here
        console.log(response.data);
      })
      .catch((error) => {
        // handle error here
        console.error(error);
      });
  };

  return (
    <div className="w-full flex flex-col gap-12 p-[72px]">
      <h1 className="font-bold text-5xl spacing leading-tight">
        Computer Vision-Based Machine Learning Model for Agricultural Plot
        Segmentation
      </h1>
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <ModelSelectField />
        <ImageUploadField handleFileChange={handleFileChange} />
        <Button type="submit">Submit</Button>
      </form>
    </div>
  );
};
