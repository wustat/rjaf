test_that("rjaf function computes correctly", {
  data(Example_data)
  
  n <- nrow(Example_data); K <- 4; gamma <- 10; sigma <- 10
  Example_trainest <- Example_data %>% slice_sample(n = floor(0.5 * nrow(Example_data)))
  Example_valid <- Example_data %>% filter(!id %in% Example_trainest$id)
  id <- "id"; y <- "Y"; trt <- "trt";  
  vars <- paste0("X", 1:3); 
  forest.reg <- rjaf(Example_trainest, Example_valid, y, id, trt, vars, clus.max = 3, 
                      clus.tree.growing = TRUE, setseed = TRUE)
  # Check if "forest.reg" has the expected class, for example, a tibble
  expect_true(is.data.frame(forest.reg), "forest.reg should be a daraframe")
})