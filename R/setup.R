#Set ggplot theme
theme_set(
  theme(
    panel.background = element_rect(color = "white"),
    panel.border = element_rect(colour = "black",
                                fill=NA,
                                linewidth = 0.25),
    panel.grid.minor=element_blank(),
    panel.grid.major = element_blank(),
    plot.background = element_blank(),
    axis.line = element_blank(),
    axis.text = element_text(color = "black"),
    text = element_text(size = 10,
                        family="Source Sans Pro",
                        color = "black"),
    axis.title = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.key = element_blank(),
    legend.key.size = (unit(3, "mm")),
    legend.position = "right",
    legend.justification = c(0, 1),
    legend.box.just = "left",
    legend.margin = margin(0, 0, 0, 0),
    legend.background = element_blank(),
    legend.box.background = element_blank(),
    title = element_text(size = 6),
    strip.background = element_blank(),
    aspect.ratio = 1
  )
)

    
    