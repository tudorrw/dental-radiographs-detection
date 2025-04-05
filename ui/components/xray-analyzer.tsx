"use client"

import { useState } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, X, ZoomIn, ZoomOut, RotateCw, Loader2, Download, Info } from "lucide-react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { Separator } from "@/components/ui/separator"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

export default function XrayAnalyzer() {
  const [files, setFiles] = useState<File[]>([])
  const [preview, setPreview] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [zoom, setZoom] = useState(100)
  const [activeTab, setActiveTab] = useState("original")

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "image/*": [".jpeg", ".jpg", ".png", ".tiff", ".dicom"],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFiles(acceptedFiles)
        const reader = new FileReader()
        reader.onload = () => {
          setPreview(reader.result as string)
          processImage(reader.result as string)
        }
        reader.readAsDataURL(acceptedFiles[0])
      }
    },
  })

  const processImage = async (imageData: string) => {
    setIsProcessing(true)

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 2000))

    // In a real application, this would call a backend API for ML processing
    // Here we're simulating the result with a mock processed image
    setProcessedImage(imageData)
    setIsProcessing(false)
  }

  const handleZoomChange = (value: number[]) => {
    setZoom(value[0])
  }

  const resetImage = () => {
    setFiles([])
    setPreview(null)
    setProcessedImage(null)
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-black">Dental X-Ray Analysis</h1>
        <p className="text-gray-600 mt-2">Upload dental X-rays for automated teeth detection and analysis</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 bg-gray-50 rounded-xl p-6 border border-gray-200">
          <h2 className="text-xl font-semibold text-black mb-4">Upload X-Ray</h2>

          <div
            {...getRootProps()}
            className={cn(
              "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
              isDragActive ? "border-cyan-500 bg-cyan-50" : "border-gray-300 hover:border-cyan-400",
            )}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
            <p className="mt-4 text-sm text-gray-600">
              Drag & drop an X-ray image, or <span className="text-cyan-600 font-medium">browse</span>
            </p>
            <p className="mt-2 text-xs text-gray-500">Supports JPEG, PNG, TIFF, and DICOM formats</p>
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
                    <p className="text-sm font-medium text-black truncate">{files[0].name}</p>
                    <p className="text-xs text-gray-500">{(files[0].size / 1024).toFixed(1)} KB</p>
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

          {preview && (
            <div className="mt-6 space-y-4">
              <h3 className="text-md font-medium text-black">Image Controls</h3>

              <div>
                <label className="text-sm text-gray-600 mb-2 block">Zoom: {zoom}%</label>
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
            </div>
          )}
        </div>

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
                      This tool automatically detects and highlights teeth in dental X-rays using AI technology.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>

          {!preview ? (
            <div className="h-[400px] flex items-center justify-center border border-gray-200 rounded-lg bg-white">
              <div className="text-center p-6">
                <div className="bg-gray-100 rounded-full p-4 inline-block mb-4">
                  <Upload className="h-8 w-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-black">No X-Ray Uploaded</h3>
                <p className="text-gray-500 mt-2 max-w-md">
                  Upload a dental X-ray image to begin analysis. The system will automatically detect and highlight
                  teeth.
                </p>
              </div>
            </div>
          ) : (
            <div>
              <Tabs defaultValue="original" value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="mb-4">
                  <TabsTrigger value="original">Original X-Ray</TabsTrigger>
                  <TabsTrigger value="processed">Processed Analysis</TabsTrigger>
                </TabsList>

                <TabsContent value="original" className="mt-0">
                  <div className="border border-gray-200 rounded-lg bg-black p-1 h-[400px] flex items-center justify-center overflow-hidden">
                    {preview && (
                      <div style={{ transform: `scale(${zoom / 100})` }} className="transition-transform duration-200">
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
                    {isProcessing ? (
                      <div className="text-center">
                        <Loader2 className="h-8 w-8 animate-spin text-cyan-500 mx-auto" />
                        <p className="text-white mt-4">Processing X-Ray...</p>
                      </div>
                    ) : processedImage ? (
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
                        {/* Simulated teeth bounding boxes */}
                        <div className="absolute top-[30%] left-[20%] w-[15%] h-[10%] border-2 border-cyan-500 rounded-sm opacity-80"></div>
                        <div className="absolute top-[30%] left-[40%] w-[15%] h-[10%] border-2 border-cyan-500 rounded-sm opacity-80"></div>
                        <div className="absolute top-[30%] left-[60%] w-[15%] h-[10%] border-2 border-cyan-500 rounded-sm opacity-80"></div>
                        <div className="absolute top-[50%] left-[25%] w-[15%] h-[10%] border-2 border-cyan-500 rounded-sm opacity-80"></div>
                        <div className="absolute top-[50%] left-[45%] w-[15%] h-[10%] border-2 border-cyan-500 rounded-sm opacity-80"></div>
                        <div className="absolute top-[50%] left-[65%] w-[15%] h-[10%] border-2 border-cyan-500 rounded-sm opacity-80"></div>
                        <div className="absolute top-[70%] left-[20%] w-[15%] h-[10%] border-2 border-cyan-500 rounded-sm opacity-80"></div>
                        <div className="absolute top-[70%] left-[40%] w-[15%] h-[10%] border-2 border-cyan-500 rounded-sm opacity-80"></div>
                        <div className="absolute top-[70%] left-[60%] w-[15%] h-[10%] border-2 border-cyan-500 rounded-sm opacity-80"></div>
                      </div>
                    ) : (
                      <div className="text-center">
                        <p className="text-white">No processed image available</p>
                      </div>
                    )}
                  </div>
                </TabsContent>
              </Tabs>

              {processedImage && activeTab === "processed" && (
                <div className="mt-4">
                  <Separator className="my-4" />
                  <h3 className="text-md font-medium text-black mb-2">Analysis Results</h3>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="text-sm font-medium text-gray-700">Detected Teeth</h4>
                      <p className="text-2xl font-bold text-black">9</p>
                      <p className="text-xs text-gray-500 mt-1">All teeth appear to be intact</p>
                    </div>

                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="text-sm font-medium text-gray-700">Potential Issues</h4>
                      <p className="text-2xl font-bold text-cyan-600">2</p>
                      <p className="text-xs text-gray-500 mt-1">Possible cavities detected</p>
                    </div>
                  </div>

                  <div className="mt-4 bg-white p-4 rounded-lg border border-gray-200">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Detailed Analysis</h4>
                    <ul className="text-sm space-y-2">
                      <li className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-cyan-500 mr-2"></div>
                        <span>Upper right molar shows potential decay (confidence: 87%)</span>
                      </li>
                      <li className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-cyan-500 mr-2"></div>
                        <span>Lower left premolar shows signs of enamel erosion (confidence: 72%)</span>
                      </li>
                      <li className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-gray-300 mr-2"></div>
                        <span>All other teeth appear normal with no significant findings</span>
                      </li>
                    </ul>
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

