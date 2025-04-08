"use client"

import { useState, useRef, useEffect } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, X, ZoomIn, ZoomOut, Move, Loader2, Download, Info, ImageIcon, ActivityIcon, RefreshCcw } from "lucide-react"
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
  const [activeTab, setActiveTab] = useState<"original" | "processed">("original")

  const [selectedModel, setSelectedModel] = useState<string>("faster-rcnn")
  const [models, setModels] = useState<string[]>(["faster-rcnn", "yolov11", "detr", "meta-model"])
  const [modelsAsString, setModelsAsString] = useState<string[]>(["Faster-RCNN", "YOLOv11", "DETR", "Meta-Model"])
  const [selectedModelName, setSelectedModelName] = useState<string>(modelsAsString[0])

  // Store detection data (or null if not processed yet)
  const [detections, setDetections] = useState<{
    boxes: number[][],
    // scores?: number[],
    labels?: number[],
    // tooth_labels?: string[],
  } | null>(null)
  
    // states for (advanced) zoom and pan functionality
    const [zoom, setZoom] = useState(100)
    const [position, setPosition] = useState({ x: 0, y: 0 })
    const [isDragging, setIsDragging] = useState(false)
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
    const imageContainerRef = useRef<HTMLDivElement>(null)
    const [isImageHovered, setIsImageHovered] = useState(false)


  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "image/*": [".jpeg", ".jpg", ".png"],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      //store the file & preview in state; NO automatic API call
      if (acceptedFiles.length > 0) {
        setFiles(acceptedFiles)

        const reader = new FileReader()
        reader.onload = () => {
          setPreview(reader.result as string)
          // Clear old processed image/detections if any
          setProcessedImage(null)
          setDetections(null)
          setActiveTab("original")
          setZoom(100)
          setPosition({ x: 0, y: 0 })
        }
        reader.readAsDataURL(acceptedFiles[0])
      }
    },
  })

  // Zoom functions
  const handleZoomChange = (value: number[]) => {
    setZoom(value[0])
  }

    const zoomIn = () => {
    setZoom((prev) => Math.min(prev + 25, 400))
  }

  const zoomOut = () => {
    setZoom((prev) => Math.max(prev - 25, 50))
  }

  const resetZoom = () => {
    setZoom(100)
    setPosition({ x: 0, y: 0 })
  }

  // Mouse event handlers for dragging/panning
  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 100) {
      setIsDragging(true)
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      })
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && zoom > 100) {
      
      const newX = e.clientX - dragStart.x
      const newY = e.clientY - dragStart.y

      console.log("previous position", dragStart.x, dragStart.y)
      console.log("new position", newX, newY)


      // Calculate boundaries to prevent dragging too far
      const container = imageContainerRef.current
      if (container) {
        const containerWidth = container.clientWidth
        const containerHeight = container.clientHeight

        // Limit the drag based on zoom level
        const maxDragX = ((zoom - 100) * containerWidth) / 200
        const maxDragY = ((zoom - 100) * containerHeight) / 200

        const boundedX = Math.max(Math.min(newX, maxDragX), -maxDragX)
        const boundedY = Math.max(Math.min(newY, maxDragY), -maxDragY)

        setPosition({ x: boundedX, y: boundedY })
      }
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  // Handle wheel events for zooming with Ctrl+scroll
  const handleWheel = (e: React.WheelEvent) => {
    // Only handle wheel events when Ctrl key is pressed
    if (e.ctrlKey) {
      // e.preventDefault()
      // Determine zoom direction
      const delta = e.deltaY < 0 ? 10 : -10

      // Calculate new zoom level
      const newZoom = Math.min(Math.max(zoom + delta, 50), 400)

      setZoom(newZoom)
    }
  }

  /**
   * call this when the user explicitly requests an analysis
   */

  useEffect(() => {
    const handleGlobalMouseUp = () => {
      setIsDragging(false)
    }
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle keyboard shortcuts when image is hovered or container is focused
      if (isImageHovered || document.activeElement === imageContainerRef.current) {
        if (e.key === "+" || e.key === "=") {
          e.preventDefault()
          zoomIn()
        } else if (e.key === "-" || e.key === "_") {
          e.preventDefault()
          zoomOut()
        } else if (e.key === "0") {
          e.preventDefault()
          resetZoom()
        }
      }
    }
    // Prevent the default browser zoom behavior on Ctrl+wheel
    const preventDefaultZoom = (e: WheelEvent) => {
      if (e.ctrlKey && isImageHovered) {
        e.preventDefault()
      }
    }

    window.addEventListener("mouseup", handleGlobalMouseUp)
    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("wheel", preventDefaultZoom, { passive: false })

    return () => {
      window.removeEventListener("mouseup", handleGlobalMouseUp)
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("wheel", preventDefaultZoom)
    }
  }, [isImageHovered, zoom])



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
      const response = await fetch(`http://localhost:8000/detections/${selectedModel}`, {
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

  const resetImage = () => {
    setFiles([])
    setPreview(null)
    setProcessedImage(null)
    setDetections(null)
    setActiveTab("original")
    resetZoom()
  }

  const resetProcessedResults = () => {
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
                {/* "X" button in case you wan to clear the image from cache */}
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
                  max={400}
                  step={5}
                  value={[zoom]}
                  onValueChange={handleZoomChange}
                  className="py-2"
                />
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="flex-1" onClick={zoomIn}>
                  <ZoomIn className="h-4 w-4 mr-2" />
                  Zoom In
                </Button>
                <Button variant="outline" size="sm" className="flex-1" onClick={zoomOut}>
                  <ZoomOut className="h-4 w-4 mr-2" />
                  Zoom Out
                </Button>
                <Button variant="outline" size="sm" className="flex-1" onClick={resetZoom}>
                  <Move className="h-4 w-4 mr-2" />
                  Reset
                </Button>
              </div>

              {/* The user must click this to call the detection API */}
              {/* The user must click this to call the detection API */}
              {!processedImage ? (
                <>
                  {/* Model selection */}
                  <div className="mt-4 mb-4">
                    <h3 className="text-md font-medium text-black mb-2">Select Detection Model</h3>
                    <div className="bg-white rounded-lg border border-gray-200 p-1">
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() =>  {
                            setSelectedModel(models[0])
                            setSelectedModelName(modelsAsString[0])  
                          }}
                          className={cn(
                            "py-2 px-3 rounded-md text-sm font-medium transition-colors",
                            selectedModel === models[0]
                              ? "bg-cyan-100 text-cyan-700 border-b-2 border-cyan-500"
                              : "text-gray-600 hover:bg-gray-100",
                          )}
                        >
                          {/* Faster-RCNN */}
                          {modelsAsString[0]}
                        </button>
                        <button
                          type="button"
                          onClick={() =>  {
                            setSelectedModel(models[1])
                            setSelectedModelName(modelsAsString[1])  
                          }}
                          className={cn(
                            "py-2 px-3 rounded-md text-sm font-medium transition-colors",
                            selectedModel === models[1]
                              ? "bg-cyan-100 text-cyan-700 border-b-2 border-cyan-500"
                              : "text-gray-600 hover:bg-gray-100",
                          )}
                        >
                          {/* YOLOv11 */}
                          {modelsAsString[1]}
                        </button>
                        <button
                          type="button"
                          onClick={() =>  {
                            setSelectedModel(models[2])
                            setSelectedModelName(modelsAsString[2])  
                          }}
                          className={cn(
                            "py-2 px-3 rounded-md text-sm font-medium transition-colors",
                            selectedModel === models[2]
                              ? "bg-cyan-100 text-cyan-700 border-b-2 border-cyan-500"
                              : "text-gray-600 hover:bg-gray-100",
                          )}
                        >
                          {/* DETR */}
                          {modelsAsString[2]} (DEtection TRansformer)
                        </button>

                        <button
                          type="button"
                          onClick={() =>  {
                            setSelectedModel(models[3])
                            setSelectedModelName(modelsAsString[3])  
                          }}
                          className={cn(
                            "py-2 px-3 rounded-md text-sm font-medium transition-colors",
                            selectedModel === models[3]
                              ? "bg-cyan-100 text-cyan-700 border-b-2 border-cyan-500"
                              : "text-gray-600 hover:bg-gray-100",
                          )}
                        >
                          {/* Meta-Model */}
                          {modelsAsString[3]} (All Models Combined)
                        </button>
                      </div>
                    </div>
                  </div>
                  <Button variant="default" onClick={handleAnalyze} disabled={isProcessing} className="w-full">
                    {isProcessing ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Analyzing with {selectedModelName}...
                      </>
                    ) : (
                      <>
                        <ActivityIcon className="h-4 w-4 mr-2" />
                        Process with {selectedModelName}
                      </>
                    )}
                  </Button>
                </>
              ) : (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-md font-medium text-black">Try Another Model</h3>

                    <div className="flex gap-2">
                      <span className="text-xs text-gray-500">Current:</span>
                      <span className="text-xs text-cyan-600 font-medium">{selectedModelName}</span>
                    </div>
                    
                  </div>
                  <Button
                    variant="outline"
                    onClick={resetProcessedResults}
                    className="w-full flex items-center justify-center"
                  >
                    <RefreshCcw className="h-4 w-4 mr-2"/>
                    Reset Analysis & Try Another Model
                  </Button>
                </div>
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
                      <br />
                      <span className="text-xs text-gray-400 mt-1 block">
                        Tip: Use Ctrl+Scroll or +/- keys to zoom. Click and drag to pan.
                      </span>

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
                  <div 
                    ref={imageContainerRef}
                    className="border border-gray-200 rounded-lg bg-black p-1 h-[400px] flex items-center justify-center overflow-hidden relative"
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseEnter={() => setIsImageHovered(true)}
                    onMouseLeave={() => setIsImageHovered(false)}
                    onWheel={handleWheel}
                    tabIndex={0} // Make the div focusable for keyboard shortcuts
                    style={{ cursor: zoom > 100 ? (isDragging ? "grabbing" : "grab") : "default" }}
                  >  
                    {preview && (
                      <div
                        style={{
                          transform: `scale(${zoom / 100})`,
                          transition: isDragging ? "none" : "transform 0.2s",
                          position: "relative",
                          left: `${position.x}px`,
                          top: `${position.y}px`,
                        }}
                      >
                        <Image
                          src={preview || "/placeholder.svg"}
                          alt="X-Ray preview"
                          width={600}
                          height={400}
                          className="max-h-[390px] w-auto"
                          draggable={false}
                        />
                      </div>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="processed" className="mt-0">
                  <div 
                    ref={imageContainerRef}
                    className="border border-gray-200 rounded-lg bg-black p-1 h-[400px] flex items-center justify-center overflow-hidden relative"
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseEnter={() => setIsImageHovered(true)}
                    onMouseLeave={() => setIsImageHovered(false)}
                    onWheel={handleWheel}
                    tabIndex={0} // Make the div focusable for keyboard shortcuts
                    style={{ cursor: zoom > 100 ? (isDragging ? "grabbing" : "grab") : "default" }}
                  >
                  
                    <div
                      style={{
                        transform: `scale(${zoom / 100})`,
                        transition: isDragging ? "none" : "transform 0.2s",
                        position: "relative",
                        left: `${position.x}px`,
                        top: `${position.y}px`,
                      }}   
                    >
                      <Image
                        src={processedImage || "/placeholder.svg"}
                        alt="Processed X-Ray"
                        width={600}
                        height={400}
                        className="max-h-[390px] w-auto"
                        draggable={false}
                      />
                    </div>
                  </div>
                </TabsContent>
              </Tabs> 

              <div className="mt-2 text-xs text-gray-500 text-center">
                <span>
                  Zoom: Ctrl+Scroll or +/- keys |{zoom > 100 ? " Click and drag to pan | " : " "}
                  Press 0 to reset view
                </span>
              </div>

            {/* If we've got detection data, show it */}
              {processedImage && detections && activeTab === "processed" && (
                <div className="mt-4">

                  <Separator className="my-4" />
                  <ToothChartSvg presentTeeth={detections?.labels} />

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
