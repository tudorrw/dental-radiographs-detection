"use client"

import { useState } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, X, ZoomIn, ZoomOut, RotateCw, Loader2, Download, Info, ImageIcon, ActivityIcon } from "lucide-react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { Separator } from "@/components/ui/separator"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { ToothChartSvg } from "./tooth-chart-svg"


export default function XrayAnalyzer() {
  const [files, setFiles] = useState<File[]>([])
  const [preview, setPreview] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [zoom, setZoom] = useState(100)
  // Keep track of which tab is active
  const [activeTab, setActiveTab] = useState<"original" | "processed">("original")
  // Store detection data (or null if not processed yet)
  const [detections, setDetections] = useState<{
    boxes: number[][],
    // scores?: number[],
    labels?: number[],
    // tooth_labels?: string[],
  } | null>(null)
  //For tooth chart

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "image/*": [".jpeg", ".jpg", ".png"],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      // 1. Just store the file & preview in state; NO automatic API call
      if (acceptedFiles.length > 0) {
        setFiles(acceptedFiles)

        const reader = new FileReader()
        reader.onload = () => {
          setPreview(reader.result as string)
          // Clear old processed image/detections if any
          setProcessedImage(null)
          setDetections(null)
          setActiveTab("original")
        }
        reader.readAsDataURL(acceptedFiles[0])
      }
    },
  })

  /**
   * Only call this when the user explicitly requests an analysis.
   */
  const handleAnalyze = async () => {
    if (!preview) return
    setIsProcessing(true)

    try {
      // Convert base64 to blob
      const byteString = atob(preview.split(",")[1])
      const mimeString = preview.split(",")[0].split(":")[1].split(";")[0]
      const ab = new ArrayBuffer(byteString.length)
      const ia = new Uint8Array(ab)

      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i)
      }

      const blob = new Blob([ab], { type: mimeString })
      const file = new File([blob], "image.jpg", { type: mimeString })

      // Create form data
      const formData = new FormData()
      formData.append("file", file)

      // Send to API
      const response = await fetch("http://localhost:8000/detections/", {
        method: "POST",
        body: formData,
      })
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`)
      }
      const result = await response.json()

      // Update UI with results
      setProcessedImage(result.processed_image)
      setDetections(result.detections || null)
      setActiveTab("processed")
    } catch (error) {
      console.error("Error processing image:", error)
      // fallback to original image if processing fails
      setProcessedImage(preview)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleZoomChange = (value: number[]) => {
    setZoom(value[0])
  }

  const resetImage = () => {
    setFiles([])
    setPreview(null)
    setProcessedImage(null)
    setDetections(null)
    setActiveTab("original")
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-black">Dental X-Ray Analysis</h1>
        <p className="text-gray-600 mt-2">
          Upload dental X-rays for automated teeth detection and analysis
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Upload Section */}
        <div className="lg:col-span-1 bg-gray-50 rounded-xl p-6 border border-gray-200">
          <h2 className="text-xl font-semibold text-black mb-4">Upload X-Ray</h2>

          <div
            {...getRootProps()}
            className={cn(
              "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
              isDragActive
                ? "border-cyan-500 bg-cyan-50"
                : "border-gray-300 hover:border-cyan-400"
            )}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
            <p className="mt-4 text-sm text-gray-600">
              Drag & drop an X-ray image, or{" "}
              <span className="text-cyan-600 font-medium">browse</span>
            </p>
            <p className="mt-2 text-xs text-gray-500">
              Supports only PNG / JPG / JPEG
            </p>
          </div>

          {files.length > 0 && (
            <div className="mt-4">
              <div className="flex items-center justify-between bg-white p-3 rounded-md border border-gray-200">
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-gray-100 rounded flex items-center justify-center">
                    <Image
                      src={preview || "/placeholder.svg"}
                      alt="Preview thumbnail"
                      width={40}
                      height={40}
                      className="object-cover rounded"
                    />
                  </div>
                  <div className="ml-3 overflow-hidden">
                    <p className="text-sm font-medium text-black truncate">
                      {files[0].name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {(files[0].size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation()
                    resetImage()
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}

          {/* If user has chosen a file, show some controls */}
          {preview && (
            <div className="mt-6 space-y-4">
              <h3 className="text-md font-medium text-black">Image Controls</h3>
              <div>
                <label className="text-sm text-gray-600 mb-2 block">
                  Zoom: {zoom}%
                </label>
                <Slider
                  defaultValue={[100]}
                  min={50}
                  max={200}
                  step={5}
                  value={[zoom]}
                  onValueChange={handleZoomChange}
                  className="py-2"
                />
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="flex-1">
                  <ZoomIn className="h-4 w-4 mr-2" />
                  Zoom In
                </Button>
                <Button variant="outline" size="sm" className="flex-1">
                  <ZoomOut className="h-4 w-4 mr-2" />
                  Zoom Out
                </Button>
                <Button variant="outline" size="sm" className="flex-1">
                  <RotateCw className="h-4 w-4 mr-2" />
                  Rotate
                </Button>
              </div>

              {/* The user must click this to call the detection API */}
              {!processedImage && (
                <Button
                  variant="default"
                  onClick={handleAnalyze}
                  disabled={isProcessing}
                  className="w-full"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <ActivityIcon className="h-4 w-4 mr-2" />
                      Process Analysis
                    </>
                  )}
                </Button>
              )}
            </div>
          )}
        </div>

        {/* Right Column: X-Ray Analysis */}
        <div className="lg:col-span-2 bg-gray-50 rounded-xl p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-black">X-Ray Analysis</h2>
            <div className="flex gap-2">
              {processedImage && (
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
              )}
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon">
                      <Info className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">
                      This tool automatically detects and highlights teeth in
                      dental X-rays using AI technology.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>

          {/* If no image is selected yet */}
          {!preview ? (
            <div className="h-[400px] flex items-center justify-center border border-gray-200 rounded-lg bg-white">
              <div className="text-center p-6">
                <div className="bg-gray-100 rounded-full p-4 inline-block mb-4">
                  <ImageIcon className="h-8 w-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-black">
                  No X-Ray Uploaded
                </h3>
                <p className="text-gray-500 mt-2 max-w-md">
                  Upload a dental X-ray image to begin analysis. The system will
                  automatically detect and highlight teeth.
                </p>
              </div>
            </div>
          ) : (
            <div>
              <Tabs
                defaultValue="original"
                value={activeTab}
                onValueChange={(v) => setActiveTab(v as typeof activeTab)}
              >
                <TabsList className="mb-4">
                  <TabsTrigger value="original">Original X-Ray</TabsTrigger>
                  {/* Only enable "Processed Analysis" tab if we have a processedImage */}
                  <TabsTrigger
                    value="processed"
                    disabled={!processedImage}
                  >
                    Processed Analysis
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="original" className="mt-0">
                  <div className="border border-gray-200 rounded-lg bg-black p-1 h-[400px] flex items-center justify-center overflow-hidden">
                    {preview && (
                      <div
                        style={{ transform: `scale(${zoom / 100})` }}
                        className="transition-transform duration-200"
                      >
                        <Image
                          src={preview || "/placeholder.svg"}
                          alt="X-Ray preview"
                          width={600}
                          height={400}
                          className="max-h-[390px] w-auto"
                        />
                      </div>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="processed" className="mt-0">
                  <div className="border border-gray-200 rounded-lg bg-black p-1 h-[400px] flex items-center justify-center overflow-hidden">
                  <div
                        style={{ transform: `scale(${zoom / 100})` }}
                        className="transition-transform duration-200 relative"
                      >
                        <Image
                          src={processedImage || "/placeholder.svg"}
                          alt="Processed X-Ray"
                          width={600}
                          height={400}
                          className="max-h-[390px] w-auto"
                        />
                      </div>
                    </div>
                </TabsContent>
              </Tabs>

            {processedImage && (
              <div className="mt-4">
                <Separator className="my-4" />
                <ToothChartSvg presentTeeth={detections?.labels} />
              </div>
            )}      

              {/* If we've got detection data, show it */}
              {processedImage && detections && activeTab === "processed" && (
                <div className="mt-4">
                  <Separator className="my-4" />
                  <h3 className="text-md font-medium text-black mb-2">
                    Detection Results
                  </h3>
                  <div className="mt-4 bg-white p-4 rounded-lg border border-gray-200">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="text-sm font-medium text-gray-700">
                        Detection Data (JSON)
                      </h4>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          // Create a downloadable JSON file
                          const dataStr =
                            "data:text/json;charset=utf-8," +
                            encodeURIComponent(
                              JSON.stringify(detections, null, 2)
                            )
                          const downloadAnchorNode =
                            document.createElement("a")
                          downloadAnchorNode.setAttribute("href", dataStr)
                          downloadAnchorNode.setAttribute(
                            "download",
                            "detection_results.json"
                          )
                          // document.body.appendChild(downloadAnchorNode)
                          downloadAnchorNode.click()
                          downloadAnchorNode.remove()
                        }}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download JSON
                      </Button>
                    </div>
                    <div className="bg-gray-50 p-3 rounded overflow-auto max-h-60 font-mono text-xs">
                      <pre>{JSON.stringify(detections, null, 2)}</pre>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
