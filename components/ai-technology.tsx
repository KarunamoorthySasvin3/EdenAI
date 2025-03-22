import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Code, Cpu, Network, Database } from "lucide-react";

export function AiTechnology() {
  return (
    <section className="w-full py-12 md:py-24 lg:py-32">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <div className="inline-block rounded-lg bg-primary/10 px-3 py-1 text-sm text-primary mb-2">
            Advanced AI Architecture
          </div>
          <div className="space-y-2">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">
              PyTorch-Powered Climate Intelligence
            </h2>
            <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
              Our garden planner uses sophisticated machine learning models to
              analyze climate data and optimize your garden's environmental
              impact.
            </p>
          </div>
        </div>

        <div className="mt-12">
          <Tabs defaultValue="models">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="models">AI Models</TabsTrigger>
              <TabsTrigger value="data">Climate Data</TabsTrigger>
              <TabsTrigger value="inference">Inference</TabsTrigger>
              <TabsTrigger value="infrastructure">Infrastructure</TabsTrigger>
            </TabsList>

            <TabsContent value="models" className="mt-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Network className="h-6 w-6 text-primary" />
                    <CardTitle>Neural Network Architecture</CardTitle>
                  </div>
                  <CardDescription>
                    Custom PyTorch models designed for ecological and climate
                    prediction
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p>
                    Our garden planner leverages several specialized PyTorch
                    models:
                  </p>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="border rounded-md p-4">
                      <h3 className="font-medium">
                        Climate Projection Transformer
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        Predicts local climate changes based on global climate
                        models
                      </p>
                      <div className="mt-2 text-xs text-primary font-mono">
                        transformer.SequenceModel()
                      </div>
                    </div>

                    <div className="border rounded-md p-4">
                      <h3 className="font-medium">Carbon Sequestration CNN</h3>
                      <p className="text-sm text-muted-foreground">
                        Estimates carbon capture based on plant selection and
                        density
                      </p>
                      <div className="mt-2 text-xs text-primary font-mono">
                        nn.Conv2d(3, 64, kernel_size=3)
                      </div>
                    </div>

                    <div className="border rounded-md p-4">
                      <h3 className="font-medium">Water Optimization RNN</h3>
                      <p className="text-sm text-muted-foreground">
                        Predicts water needs based on weather patterns and plant
                        types
                      </p>
                      <div className="mt-2 text-xs text-primary font-mono">
                        nn.LSTM(input_size, hidden_size)
                      </div>
                    </div>

                    <div className="border rounded-md p-4">
                      <h3 className="font-medium">
                        Biodiversity Enhancement GNN
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        Graph neural network for optimizing ecosystem
                        relationships
                      </p>
                      <div className="mt-2 text-xs text-primary font-mono">
                        gnn.MessagePassing()
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="data" className="mt-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Database className="h-6 w-6 text-primary" />
                    <CardTitle>Climate Data Sources</CardTitle>
                  </div>
                  <CardDescription>
                    Comprehensive datasets used to train our AI models
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p>
                    Our models are trained on diverse climate and ecological
                    datasets:
                  </p>

                  <ul className="space-y-2">
                    <li className="flex items-start gap-2">
                      <span className="bg-primary/10 text-primary rounded-full px-2 py-0.5 text-xs">
                        IPCC
                      </span>
                      <span>
                        Global climate projections from the Intergovernmental
                        Panel on Climate Change
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="bg-primary/10 text-primary rounded-full px-2 py-0.5 text-xs">
                        GBIF
                      </span>
                      <span>
                        Global Biodiversity Information Facility's species
                        distribution data
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="bg-primary/10 text-primary rounded-full px-2 py-0.5 text-xs">
                        FLUXNET
                      </span>
                      <span>
                        Carbon flux measurements from ecosystems worldwide
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="bg-primary/10 text-primary rounded-full px-2 py-0.5 text-xs">
                        GLDAS
                      </span>
                      <span>
                        Global Land Data Assimilation System for soil moisture
                        and temperature
                      </span>
                    </li>
                  </ul>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="inference" className="mt-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Cpu className="h-6 w-6 text-primary" />
                    <CardTitle>Real-time Inference</CardTitle>
                  </div>
                  <CardDescription>
                    How our PyTorch models analyze your garden in real-time
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p>
                    Our platform provides dynamic garden recommendations
                    through:
                  </p>

                  <div className="space-y-4">
                    <div className="border-l-4 border-primary pl-4 py-2">
                      <h3 className="font-medium">TorchScript Optimization</h3>
                      <p className="text-sm text-muted-foreground">
                        Pre-compiled models for efficient inference on both
                        cloud and edge devices
                      </p>
                    </div>

                    <div className="border-l-4 border-primary pl-4 py-2">
                      <h3 className="font-medium">Transfer Learning</h3>
                      <p className="text-sm text-muted-foreground">
                        Models fine-tuned to your specific microclimate and
                        garden conditions
                      </p>
                    </div>

                    <div className="border-l-4 border-primary pl-4 py-2">
                      <h3 className="font-medium">Continuous Integration</h3>
                      <p className="text-sm text-muted-foreground">
                        Models updated with latest climate data and plant
                        performance metrics
                      </p>
                    </div>

                    <div className="border-l-4 border-primary pl-4 py-2">
                      <h3 className="font-medium">
                        Uncertainty Quantification
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        Bayesian techniques to provide confidence intervals on
                        climate predictions
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="infrastructure" className="mt-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Code className="h-6 w-6 text-primary" />
                    <CardTitle>Technical Infrastructure</CardTitle>
                  </div>
                  <CardDescription>
                    Behind-the-scenes technology powering our climate-focused
                    garden planner
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <h3 className="font-medium">PyTorch Ecosystem</h3>
                      <ul className="space-y-1 text-sm">
                        <li>
                          • PyTorch Lightning for structured model training
                        </li>
                        <li>• TorchServe for model deployment</li>
                        <li>• TorchVision for plant image analysis</li>
                        <li>
                          • PyTorch Geometric for ecological relationships
                        </li>
                      </ul>
                    </div>

                    <div className="space-y-2">
                      <h3 className="font-medium">Distributed Computing</h3>
                      <ul className="space-y-1 text-sm">
                        <li>• Ray clusters for parallel inference</li>
                        <li>• GPU acceleration for climate simulations</li>
                        <li>• Distributed training on climate datasets</li>
                        <li>• Edge deployment for local device inference</li>
                      </ul>
                    </div>
                  </div>

                  <div className="mt-6 p-4 bg-muted rounded-md">
                    <h3 className="font-medium mb-2">Carbon-Aware Computing</h3>
                    <p className="text-sm">
                      Our infrastructure is designed to minimize carbon
                      footprint:
                    </p>
                    <ul className="mt-2 space-y-1 text-xs">
                      <li>
                        • Models trained on renewable energy-powered compute
                        clusters
                      </li>
                      <li>• Model quantization to reduce computation needs</li>
                      <li>
                        • Efficient model pruning to minimize power consumption
                      </li>
                      <li>
                        • Carbon-aware training schedules aligned with grid's
                        cleanest energy
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </section>
  );
}
