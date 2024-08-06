"use client";
import React, { useState } from "react";
import { toast } from "@/components/ui/use-toast";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, FormProvider } from "react-hook-form";
import { ImagePreview } from "@/components/image-preview";
import { formSchema, FormData } from "@/utils/form-schema";
import { FormContainer } from "@/components/form-container";

export default function Home() {
  const form = useForm<FormData>({
    resolver: zodResolver(formSchema),
  });

  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const onSubmit = (data: FormData) => {
    toast({
      title: "Form submitted",
      description: JSON.stringify(data, null, 2),
    });
  };

  const handleFileChange = (file: File | null) => {
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setUploadedImage(null);
    }
  };

  return (
    <main className="flex w-svw h-svh">
      <FormProvider {...form}>
        <FormContainer
          onSubmit={onSubmit}
          handleFileChange={handleFileChange}
        />
      </FormProvider>
      <ImagePreview uploadedImage={uploadedImage} />
    </main>
  );
}
