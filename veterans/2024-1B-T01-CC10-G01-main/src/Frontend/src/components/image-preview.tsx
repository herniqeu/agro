"use client";
import React from "react";
import {
  ImageOff,
  ScanLine,
  Satellite,
  Laptop2,
  LayoutDashboard,
} from "lucide-react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

type ImagePreviewProps = {
  uploadedImage: string | null;
};

export const ImagePreview: React.FC<ImagePreviewProps> = ({
  uploadedImage,
}) => (
  <div className="min-w-[1080px] h-full bg-[#F5F5F5] flex items-center justify-center relative">
    {uploadedImage ? (
      <img
        src={uploadedImage}
        alt="Uploaded"
        className="absolute inset-0 w-full h-full object-cover"
      />
    ) : (
      <ImageOff color="#D4D4D4" size={72} strokeWidth={1.5} />
    )}

    <div className="absolute top-6 left-6 flex flex-col gap-2">
      <ToggleGroup
        type="single"
        className="flex flex-col bg-white p-[6px] rounded-lg"
      >
        <ToggleGroupItem value="scan" aria-label="Toggle scan">
          <ScanLine className="h-4 w-4" />
        </ToggleGroupItem>
      </ToggleGroup>
      <ToggleGroup
        type="single"
        className="flex flex-col bg-white p-[6px] rounded-lg"
      >
        <ToggleGroupItem value="satellite" aria-label="Toggle satellite">
          <Satellite className="h-4 w-4" />
        </ToggleGroupItem>
        <ToggleGroupItem value="laptop" aria-label="Toggle laptop">
          <Laptop2 className="h-4 w-4" />
        </ToggleGroupItem>
        <ToggleGroupItem value="dashboard" aria-label="Toggle dashboard">
          <LayoutDashboard className="h-4 w-4" />
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  </div>
);
