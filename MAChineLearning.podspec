Pod::Spec.new do |s|

  # ―――  Spec Metadata  ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  s.name         = "MAChineLearning"
  s.version      = "1.0.0"
  s.summary      = "Machine Learning for the Mac."

  s.description  = <<-DESC
                   MAChineLearning is framework that provides a quick and easy way to experiment Machine Learning with
                   native code on the Mac, with some specific support for Natural Language Processing. It is written in Objective-C, but it is
                   compatible by Swift.

                   Currently the framework supports:

                   * Neural Networks
                   * Bag of Words
                   * Word Vectors

                   Differently than many other machine learning libraries for macOS and iOS, MAChineLearning includes full training implementation
                   for its neural networks. You don't need a separate language or another framework to train the network, you have all you need here.
                   DESC

  s.homepage     = "https://github.com/gianlucabertani/MAChineLearning"


  # ―――  Spec License  ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  s.license      = { :type => 'BSD 2.0', :file => 'LICENSE' }


  # ――― Author Metadata  ――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  s.author             = { "Gianluca Bertani" => "gianluca.bertani@email.it" }
  s.social_media_url    = "https://twitter.com/self_vs_this"


  # ――― Platform Specifics ――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  s.platform     = :osx, "10.9"


  # ――― Source Location ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  s.source       = { :git => "https://github.com/gianlucabertani/MAChineLearning.git",
                     :tag => s.version.to_s }


  # ――― Source Code ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  s.source_files  = "MAChineLearning/**/*.{h,m}"


  # ――― Project Linking ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  s.frameworks  = "Accelerate"


  # ――― Project Settings ――――――――――――――――――――――――――――――――――――――――――――――――――――――――― #

  s.requires_arc = true

end
