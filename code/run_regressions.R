# Estimate four dialogue-proxy models from the analysis CSV.
options(stringsAsFactors=FALSE, width=180, digits=17)

MEASURES <- c('E_baseline', 'E_learning_observed', 'E_net_observed', 'E_reasoning_observed')
LABELS <- c(
  E_baseline='Understanding share among classified problems',
  E_learning_observed='LLM learning evidence / observed problems',
  E_net_observed='Net U-minus-A evidence / observed problems',
  E_reasoning_observed='Specific reasoning-help coverage R/O')
CONTROLS <- c('T','D','G','grade10','grade11','session_id2','session_id3','session_id4',
  'teacher_idteacher_10','teacher_idteacher_12','teacher_idteacher_13','teacher_idteacher_14',
  'teacher_idteacher_15','teacher_idteacher_16','teacher_idteacher_17','teacher_idteacher_18',
  'teacher_idteacher_19','teacher_idteacher_2','teacher_idteacher_4','teacher_idteacher_5',
  'teacher_idteacher_6','teacher_idteacher_7','teacher_idteacher_8','teacher_idteacher_9',
  'grader_id1&2','grader_id2','grader_id3','grader_id4','grader_id5','grader_id6',
  'grader_id7','grader_id8','grader_id9&10')

require_ok <- function(condition, message) {
  if(length(condition)!=1L || is.na(condition) || !condition) stop(message, call.=FALSE)
}

stats_row <- function(estimate, se, df) {
  require_ok(all(is.finite(c(estimate,se,df))) && se>0 && df>0,
             'Invalid coefficient, standard error or degrees of freedom.')
  critical <- qt(.975,df)
  list(estimate=as.numeric(estimate), se_CR1=as.numeric(se),
       p_t=2*pt(-abs(estimate/se),df), ci95_low=estimate-critical*se,
       ci95_high=estimate+critical*se)
}

ols <- function(x, y, cluster_index, groups) {
  n <- nrow(x); k <- ncol(x)
  decomposition <- qr(x,tol=1e-10)
  require_ok(decomposition$rank==k, 'A regression design is rank deficient.')
  coefficients <- as.numeric(qr.coef(decomposition,y))
  residual <- as.numeric(y-x%*%coefficients)
  inverse <- backsolve(qr.R(decomposition),diag(k))
  bread <- tcrossprod(inverse)
  scores <- rowsum(x*residual,cluster_index,reorder=TRUE)
  correction <- groups/(groups-1)*(n-1)/(n-k)
  influence <- scores%*%bread*sqrt(correction)
  list(b=coefficients,residual=residual,influence=influence,
       covariance=crossprod(influence),bread=bread)
}

cr2_theta_basis <- function(x, treatment_index, cluster_index, groups) {
  # Full-Q leverage adjustment: A_g=(I-H_gg)^{+(-1/2)}.
  # v_g=(I-H)E_g A_g ell_g; se=sqrt(sum_g (v_g'y)^2).
  # iid working-model Satterthwaite df=tr(V'V)^2/tr((V'V)^2).
  n <- nrow(x)
  other <- x[,-treatment_index,drop=FALSE]
  residual_t <- qr.resid(qr(other,tol=1e-10),x[,treatment_index])
  ell <- as.numeric(residual_t/sum(residual_t^2))
  basis <- qr.Q(qr(x,tol=1e-10))
  vectors <- matrix(0,n,groups)
  null_norms <- minimum_eigenvalues <- numeric(groups)
  nullities <- integer(groups)
  for(group in seq_len(groups)) {
    index <- which(cluster_index==group)
    block <- basis[index,,drop=FALSE]
    residual_block <- diag(length(index))-tcrossprod(block)
    residual_block <- (residual_block+t(residual_block))/2
    eig <- eigen(residual_block,symmetric=TRUE)
    require_ok(min(eig$values)>=-1e-9, 'A full-design leverage block is not positive semidefinite.')
    positive <- eig$values>1e-10
    null_norm <- sqrt(sum(crossprod(eig$vectors[,!positive,drop=FALSE],ell[index])^2))
    require_ok(null_norm<=1e-9*max(1,sqrt(sum(ell[index]^2))),
               'The treatment contrast is not orthogonal to a singular leverage direction.')
    adjusted <- as.numeric(eig$vectors[,positive,drop=FALSE]%*%
      (crossprod(eig$vectors[,positive,drop=FALSE],ell[index])/sqrt(eig$values[positive])))
    vectors[index,group] <- adjusted
    vectors[,group] <- vectors[,group]-as.numeric(basis%*%crossprod(block,adjusted))
    null_norms[group] <- null_norm
    minimum_eigenvalues[group] <- min(eig$values)
    nullities[group] <- sum(!positive)
  }
  gram <- crossprod(vectors); trace <- sum(diag(gram)); target <- sum(ell^2)
  require_ok(abs(trace-target)<=1e-12+1e-8*abs(target), 'The CR2 iid-working-model trace check failed.')
  df <- trace^2/sum(gram^2)
  require_ok(is.finite(df) && df>0, 'The CR2 Satterthwaite degrees of freedom are invalid.')
  list(vectors=vectors,df=df,metadata=list(
    working_model='iid working covariance for CR2/Satterthwaite',
    full_design_used=TRUE,eigenvalue_tolerance=1e-10,
    cluster_leverage_nullities=nullities,maximum_null_space_contrast_norm=max(null_norms),
    minimum_residual_block_eigenvalue=min(minimum_eigenvalues),trace_VtV=trace,
    OLS_contrast_squared_norm=target,trace_check_absolute_error=abs(trace-target),
    satterthwaite_df=df,CR1_multiplier_applied_to_CR2=FALSE,gamma_CR2_computed=FALSE))
}

quadratic_confidence_set <- function(a,b,c) {
  # Solve a*r^2+b*r+c <= 0 analytically, including unbounded/disjoint sets.
  require_ok(all(is.finite(c(a,b,c))), 'Nonfinite Fieller polynomial.')
  scale <- max(abs(c(a,b,c)))
  if(scale>0) { a <- a/scale; b <- b/scale; c <- c/scale }
  result <- function(kind,intervals) list(kind=kind,intervals=intervals)
  interval <- function(low,high) matrix(c(low,high),ncol=2)
  if(a==0) {
    if(b==0) return(if(c<=0) result('entire real line',interval(-Inf,Inf)) else result('empty set',matrix(numeric(0),ncol=2)))
    root <- -c/b
    return(if(b>0) result('upper half-line',interval(-Inf,root)) else result('lower half-line',interval(root,Inf)))
  }
  discriminant <- b^2-4*a*c
  if(discriminant<0) return(if(a<0) result('entire real line',interval(-Inf,Inf)) else result('empty set',matrix(numeric(0),ncol=2)))
  # Stable quadratic roots avoid subtracting nearly equal numbers.
  q <- -.5*(b+if(b>=0) sqrt(discriminant) else -sqrt(discriminant))
  roots <- if(q==0) rep(-b/(2*a),2) else sort(c(q/a,c/q))
  if(a>0) result('bounded interval',interval(roots[1],roots[2])) else
    if(discriminant==0) result('entire real line',interval(-Inf,Inf)) else
      result('two disjoint half-lines',rbind(c(-Inf,roots[1]),c(roots[2],Inf)))
}

format_confidence_set <- function(set, number_format='%.5f') {
  if(set$kind %in% c('entire real line','empty set')) return(set$kind)
  paste(apply(set$intervals,1,function(bounds) {
    left <- if(is.infinite(bounds[1])) '(-Inf' else paste0('[',sprintf(number_format,bounds[1]))
    right <- if(is.infinite(bounds[2])) 'Inf)' else paste0(sprintf(number_format,bounds[2]),']')
    paste0(left,', ',right)
  }),collapse=' union ')
}

paper_table <- function(comparison) {
  number <- function(x,digits=5) ifelse(is.na(x),'-',sprintf(paste0('%.',digits,'f'),x))
  significance_stars <- function(p) {
    require_ok(all(is.na(p) | (is.finite(p) & p>=0 & p<=1)),
      'Table significance stars require valid p-values or NA.')
    ifelse(is.na(p),'',ifelse(p<.01,'***',ifelse(p<.05,'**',ifelse(p<.10,'*',''))))
  }
  # Exact boundaries and missing values: stars use full-precision p-values,
  # never rounded display values or a coefficient/SE reconstruction.
  require_ok(identical(
    significance_stars(c(0,.009999,.01,.049999,.05,.099999,.10,NA_real_)),
    c('***','***','**','**','*','*','','')),
    'Table significance-star threshold self-check failed.')
  coefficient <- function(value,p) {
    ifelse(is.na(value),'-',paste0(number(value),significance_stars(p)))
  }
  standard_error <- function(value) ifelse(is.na(value),'-',paste0('(',number(value),')'))
  pvalue <- function(x) vapply(x,function(value) {
    if(is.na(value)) '-' else if(value<.0001) sprintf('%.3g',value) else sprintf('%.4f',value)
  },character(1))
  interval <- function(low,high) sprintf('[%.5f, %.5f]',low,high)
  definitions <- list(
    list('Construction',comparison$label),
    list('theta',coefficient(comparison$theta_estimate,comparison$theta_p_CR2)),
    list('Theta SE: four-column CR2',standard_error(comparison$theta_se_CR2)),
    list('Theta p: CR2/Satterthwaite',pvalue(comparison$theta_p_CR2)),
    list('Theta 95% CI: CR2',interval(comparison$theta_ci95_CR2_low,comparison$theta_ci95_CR2_high)),
    list('Theta p: CR2, Holm (3 alternatives)',pvalue(comparison$theta_p_holm3_CR2)),
    list('Theta SE: classroom CR1',standard_error(comparison$theta_se_CR1)),
    list('Theta p: CR1, t(26)',pvalue(comparison$theta_p_t)),
    list('rho',coefficient(comparison$rho_estimate,comparison$rho_p_t)),
    list('Rho SE: classroom CR1 delta',standard_error(comparison$rho_se_CR1)),
    list('Rho p: CR1 delta, two-sided',pvalue(comparison$rho_p_t)),
    list('Rho 95% CI: CR1 delta',interval(comparison$rho_ci95_low,comparison$rho_ci95_high)),
    list('Rho Fieller 95% set',comparison$rho_Fieller95_confidence_set),
    list('gamma',coefficient(comparison$gamma_estimate,comparison$gamma_p_t)),
    list('Gamma SE: classroom CR1 delta',standard_error(comparison$gamma_se_CR1)),
    list('Gamma p: CR1 delta, two-sided',pvalue(comparison$gamma_p_t)),
    list('Gamma 95% CI: CR1 delta',interval(comparison$gamma_ci95_low,comparison$gamma_ci95_high)),
    list('Mediated score component',coefficient(comparison$mediated_estimate,comparison$mediated_p_t)),
    list('Mediated SE: classroom CR1 delta',standard_error(comparison$mediated_se_CR1)),
    list('Mediated p: CR1 delta, two-sided',pvalue(comparison$mediated_p_t)),
    list('Total score contrast',coefficient(comparison$total_effect_estimate,comparison$total_effect_p_t)),
    list('Total score SE: classroom CR1',standard_error(comparison$total_effect_se_CR1)),
    list('Total score p: CR1',pvalue(comparison$total_effect_p_t)),
    list('Excluded-GPA first-stage F: CR1',number(comparison$first_stage_F,3)),
    list('First-stage partial R2',number(comparison$partial_R2)),
    list('Observations',as.character(comparison$n)),
    list('Classrooms',as.character(comparison$classrooms)))
  result <- data.frame(row=vapply(definitions,`[[`,character(1),1),check.names=FALSE)
  for(j in seq_along(MEASURES)) result[[MEASURES[j]]] <- vapply(definitions,function(item) item[[2]][j],character(1))
  result
}

latex_escape <- function(text) {
  replacements <- setNames(c(r"(\textbackslash{})",r"(\&)",r"(\%)",r"(\$)",
                             r"(\#)",r"(\_)",r"(\{)",r"(\})"),c('\\','&','%','$','#','_','{','}'))
  paste0(vapply(strsplit(as.character(text),'',fixed=TRUE)[[1]],function(ch)
    if(ch %in% names(replacements)) replacements[[ch]] else ch,character(1)),collapse='')
}

table_latex <- function(table) {
  labels <- c('theta'=r"($\theta$)",'rho'=r"($\rho$)",'gamma'=r"($\gamma$)")
  display_value <- function(value) {
    stars <- regmatches(value,regexpr('[*]+$',value))
    if(length(stars)==1L && nzchar(stars)) {
      paste0('$',sub('[*]+$','',value),'^{',stars,'}$')
    } else latex_escape(value)
  }
  lines <- c(r"(\begin{table}[htbp])",r"(\centering)",
    r"(\caption{Dialogue patterns and the GPA-instrumented decomposition})",
    r"(\label{tab:dialogue-decomposition})",r"(\begingroup)",r"(\small)",
    r"(\setlength{\tabcolsep}{4pt})",r"(\renewcommand{\arraystretch}{1.1})",
    r"(\resizebox{\linewidth}{!}{\begin{tabular}{lcccc})",r"(\hline)",
    r"( & Understanding & LLM evidence & Net U--A & Reasoning \\)",r"(\hline)")
  for(i in seq_len(nrow(table))) {
    label <- table$row[i]
    if(label=='Construction') next
    values <- unlist(table[i,MEASURES],use.names=FALSE)
    display_label <- if(label %in% names(labels)) labels[[label]] else latex_escape(label)
    lines <- c(lines,paste0(display_label,' & ',paste(vapply(values,display_value,character(1)),collapse=' & '),r"( \\)"))
  }
  c(lines,r"(\hline)",r"(\end{tabular}})",r"(\endgroup)",r"(\par\medskip)",
    r"(\parbox{\linewidth}{\footnotesize Notes: Standard errors are in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$, using unrounded two-sided p-values. Theta stars and the main theta standard error use CR2/Satterthwaite for the four-column dialogue design; diagnostic CR1 rows are retained. Rho, gamma, and the mediated component use joint cross-equation classroom CR1 delta inference with $t(26)$; the total contrast uses classroom CR1 inference. All columns use 1,163 student-sessions and 27 classrooms. Fieller sets invert the joint reduced-form GPA-coefficient contrast. Holm adjusts the three alternative-measure theta tests; displayed stars use the unadjusted p-values.})",
    r"(\end{table})")
}

table_markdown <- function(table) {
  clean <- function(value) gsub('\n',' ',gsub('|','/',as.character(value),fixed=TRUE),fixed=TRUE)
  lines <- c(paste0('| ',paste(clean(names(table)),collapse=' | '),' |'),
             paste0('| ',paste(rep('---',ncol(table)),collapse=' | '),' |'))
  for(i in seq_len(nrow(table))) lines <- c(lines,paste0('| ',paste(clean(unlist(table[i,],use.names=FALSE)),collapse=' | '),' |'))
  c(lines,'',
    'Standard errors are in parentheses. Significance: *** p < 0.01; ** p < 0.05; * p < 0.10. Stars use unrounded two-sided p-values.',
    'Theta stars and its main standard error use four-column CR2/Satterthwaite; diagnostic CR1 rows are retained. Rho, gamma, and the mediated component use joint classroom CR1 delta inference, t(26). The total contrast uses classroom CR1 inference.',
    'Fieller 95% sets invert the joint reduced-form GPA-coefficient contrast. Holm adjusts the three alternative-measure theta tests; displayed stars use the unadjusted p-values.')
}

scaled_error <- function(actual, expected) {
  max(abs(actual-expected))/max(1,abs(actual),abs(expected))
}

fieller_inference <- function(pi_EG, pi_YG, influence_EG, influence_YG, df) {
  variance_EG <- sum(influence_EG^2)
  variance_YG <- sum(influence_YG^2)
  covariance <- sum(influence_EG*influence_YG)
  critical_squared <- qt(.975,df)^2
  a <- pi_EG^2-critical_squared*variance_EG
  b <- -2*pi_EG*pi_YG+2*critical_squared*covariance
  c <- pi_YG^2-critical_squared*variance_YG
  set <- quadratic_confidence_set(a,b,c)
  rho <- pi_YG/pi_EG

  # Invert the joint GPA-coefficient contrast from both fitted equations.
  candidates <- unique(c(-100,-10,-1,0,1,10,100,rho))
  polynomial_error <- max(vapply(candidates,function(r) {
    direct <- (pi_YG-r*pi_EG)^2-
      critical_squared*sum((influence_YG-r*influence_EG)^2)
    polynomial <- a*r^2+b*r+c
    scaled_error(direct,polynomial)
  },numeric(1)))
  endpoints <- as.numeric(set$intervals)
  endpoints <- endpoints[is.finite(endpoints)]
  endpoint_error <- if(length(endpoints)) max(vapply(endpoints,function(r) {
    abs(a*r^2+b*r+c)/max(1,abs(a*r^2),abs(b*r),abs(c))
  },numeric(1))) else 0
  includes_estimate <- nrow(set$intervals)>0 && any(
    rho>=set$intervals[,1] & rho<=set$intervals[,2])
  require_ok(polynomial_error<1e-10 && endpoint_error<1e-9 && includes_estimate,
    'The separate-design Fieller inversion checks failed.')
  list(fields=list(
    rho_Fieller95_set_type=set$kind,
    rho_Fieller95_confidence_set=format_confidence_set(set),
    rho_Fieller95_set_full=format_confidence_set(set,'%.17g'),
    rho_Fieller95_strictly_positive=nrow(set$intervals)>0 && all(set$intervals[,1]>0),
    rho_Fieller_p_zero=2*pt(-abs(pi_YG/sqrt(variance_YG)),df),
    rho_Fieller_qa=a,rho_Fieller_qb=b,rho_Fieller_qc=c,
    rho_Fieller_discriminant=b^2-4*a*c,
    rho_RF_G_cross_covariance_CR1=covariance),
    checks=list(polynomial_scaled_error=polynomial_error,
      endpoint_scaled_error=endpoint_error,point_estimate_in_confidence_set=includes_estimate))
}

estimate_models <- function(analysis) {
  required <- unique(c('student_id','class_id','session_id','Y',CONTROLS,MEASURES))
  require_ok(all(required %in% names(analysis)),paste('Model columns missing:',
    paste(setdiff(required,names(analysis)),collapse=', ')))
  require_ok(!anyDuplicated(analysis[c('student_id','class_id','session_id')]),
    'Duplicate model student-class-session keys.')
  numeric_columns <- unique(c('Y',CONTROLS,MEASURES))
  require_ok(all(vapply(analysis[numeric_columns],is.numeric,logical(1))) &&
    all(is.finite(as.matrix(analysis[numeric_columns]))),
    'Model regressions require complete finite numeric columns.')
  require_ok(nrow(analysis)==1163 && length(unique(analysis$student_id))==468,
    'The model common sample must contain 1163 sessions and 468 students.')
  require_ok(!anyNA(analysis[c('student_id','class_id','session_id')]),
    'A model identifier is missing.')

  x_E <- cbind('(Intercept)'=1,as.matrix(analysis[c('T','D','G')]))
  x_Y <- cbind('(Intercept)'=1,as.matrix(analysis[CONTROLS]))
  n <- nrow(x_E); k_E <- ncol(x_E); k_Y <- ncol(x_Y)
  labels <- sort(unique(as.character(analysis$class_id)),method='radix')
  cluster_index <- match(as.character(analysis$class_id),labels)
  groups <- length(labels); df <- groups-1
  require_ok(k_E==4 && k_Y==34 && groups==27 && all(analysis$T %in% c(0,1)),
    'Unexpected model design dimensions, classrooms or treatment coding.')
  require_ok(all(vapply(split(analysis$T,cluster_index),function(value)
    length(unique(value))==1L,logical(1))), 'Treatment varies within a classroom.')
  require_ok(qr(x_E,tol=1e-10)$rank==k_E && qr(x_Y,tol=1e-10)$rank==k_Y,
    'A model reduced-form design is rank deficient.')
  require_ok(!anyDuplicated(colnames(x_E)) && !anyDuplicated(colnames(x_Y)),
    'A model design has duplicate column names.')
  # Treatment and GPA indices for each equation.
  t_E <- match('T',colnames(x_E)); g_E <- match('G',colnames(x_E))
  t_Y <- match('T',colnames(x_Y)); g_Y <- match('G',colnames(x_Y))
  embedding <- match(colnames(x_E),colnames(x_Y))
  require_ok(!anyNA(c(t_E,g_E,t_Y,g_Y,embedding)), 'A model design index is missing.')
  W <- setdiff(colnames(x_Y),c('(Intercept)','T','D','G'))
  correction_E <- groups/(groups-1)*(n-1)/(n-k_E)
  correction_Y <- groups/(groups-1)*(n-1)/(n-k_Y)
  fy <- ols(x_Y,analysis$Y,cluster_index,groups)
  cr2 <- cr2_theta_basis(x_E,t_E,cluster_index,groups)
  bread_E <- solve(crossprod(x_E)); bread_Y <- solve(crossprod(x_Y))
  score_Y <- rowsum(x_Y*fy$residual,cluster_index,reorder=TRUE)
  normal_Y_error <- max(abs(crossprod(x_Y,fy$residual)))/
    max(1,abs(crossprod(x_Y,analysis$Y)))
  reference_Y <- as.numeric(bread_Y%*%crossprod(x_Y,analysis$Y))
  coefficient_Y_error <- scaled_error(fy$b,reference_Y)
  require_ok(normal_Y_error<1e-9 && coefficient_Y_error<1e-8,
    'Independent score-regression checks failed.')

  rows <- reduced <- diagnostics <- list()
  for(m in seq_along(MEASURES)) {
    measure <- MEASURES[m]; proxy <- analysis[[measure]]
    require_ok(sd(proxy)>1e-12,paste('A model proxy is constant:',measure))
    fe <- ols(x_E,proxy,cluster_index,groups)
    be <- fe$b; by <- fy$b; he <- fe$influence; hy <- fy$influence
    require_ok(abs(be[g_E])>1e-12,paste('The restricted GPA first stage is zero:',measure))
    theta <- be[t_E]; total <- by[t_Y]; rho <- by[g_Y]/be[g_E]
    h_rho <- (hy[,g_Y]-rho*he[,g_E])/be[g_E]
    h_mediated <- rho*he[,t_E]+theta*h_rho
    h_gamma <- hy[,t_Y]-h_mediated
    terms <- list(theta=c(theta,sqrt(sum(he[,t_E]^2))),
      rho=c(rho,sqrt(sum(h_rho^2))),
      gamma=c(total-rho*theta,sqrt(sum(h_gamma^2))),
      mediated=c(rho*theta,sqrt(sum(h_mediated^2))),
      total_effect=c(total,sqrt(sum(hy[,t_Y]^2))))
    residual_without_G <- qr.resid(qr(x_E[,-g_E,drop=FALSE],tol=1e-10),proxy)
    first_stage_F <- be[g_E]^2/fe$covariance[g_E,g_E]
    row <- list(measure_id=measure,label=unname(LABELS[[measure]]),
      specification='equation_specific_reduced_forms',n=n,students=468,classrooms=groups,
      rank_RF_E=k_E,rank_RF_Y=k_Y,df_CR1=df,
      CR1_multiplier_RF_E=correction_E,CR1_multiplier_RF_Y=correction_Y,
      piET=theta,piEG=be[g_E],piYT=total,piYG=by[g_Y],
      first_stage_F=first_stage_F,
      first_stage_p=pf(first_stage_F,1,df,lower.tail=FALSE),
      partial_R2=1-sum(fe$residual^2)/sum(residual_without_G^2),
      mean_E=mean(proxy),sd_E=sd(proxy))
    for(term in names(terms)) {
      values <- stats_row(terms[[term]][1],terms[[term]][2],df)
      for(field in names(values)) row[[paste(term,field,sep='_')]] <- values[[field]]
    }
    fieller <- fieller_inference(be[g_E],by[g_Y],he[,g_E],hy[,g_Y],df)
    row <- c(row,fieller$fields)
    row$rho_p_greater_CR1 <- pt(rho/row$rho_se_CR1,df,lower.tail=FALSE)
    row$rho_lower95_one_sided_CR1 <- rho-qt(.95,df)*row$rho_se_CR1
    row$rho_significant_positive_CR1 <- row$rho_p_greater_CR1<.05
    cr2_se <- sqrt(sum(crossprod(cr2$vectors,proxy)^2))
    require_ok(is.finite(cr2_se) && cr2_se>0,'The model theta CR2 standard error is invalid.')
    row$theta_se_CR2 <- cr2_se; row$theta_df_CR2 <- cr2$df
    row$theta_p_CR2 <- 2*pt(-abs(theta/cr2_se),cr2$df)
    row$theta_p_greater_CR2 <- pt(theta/cr2_se,cr2$df,lower.tail=FALSE)
    row$theta_ci95_CR2_low <- theta-qt(.975,cr2$df)*cr2_se
    row$theta_ci95_CR2_high <- theta+qt(.975,cr2$df)*cr2_se

    # Restricted-design predictions and joint classroom covariance.
    embedded <- setNames(rep(0,k_Y),colnames(x_Y)); embedded[embedding] <- be
    embedding_error <- max(abs(as.numeric(x_Y%*%embedded)-as.numeric(x_E%*%be)))
    normal_E_error <- max(abs(crossprod(x_E,fe$residual)))/max(1,abs(crossprod(x_E,proxy)))
    coefficient_E_error <- scaled_error(be,as.numeric(bread_E%*%crossprod(x_E,proxy)))
    score_E <- rowsum(x_E*fe$residual,cluster_index,reorder=TRUE)
    joint_influence <- cbind(he,hy)
    joint_covariance <- crossprod(joint_influence)
    reference_cross <- sqrt(correction_E*correction_Y)*
      bread_E%*%crossprod(score_E,score_Y)%*%bread_Y
    cross_error <- scaled_error(crossprod(he,hy),reference_cross)
    reference_E <- correction_E*bread_E%*%crossprod(score_E)%*%bread_E
    reference_Y_cov <- correction_Y*bread_Y%*%crossprod(score_Y)%*%bread_Y
    covariance_error <- max(scaled_error(fe$covariance,reference_E),
      scaled_error(fy$covariance,reference_Y_cov))
    symmetry_error <- max(abs(joint_covariance-t(joint_covariance)))
    eigenvalues <- eigen(joint_covariance,symmetric=TRUE,only.values=TRUE)$values
    minimum_eigenvalue <- min(eigenvalues)
    require_ok(minimum_eigenvalue>=-1e-10*max(1,max(abs(eigenvalues))),
      'The joint model CR1 covariance is not positive semidefinite.')

    # Independent four-coefficient gradient check, including every covariance.
    rf_coefficients <- c(be[t_E],be[g_E],by[t_Y],by[g_Y])
    transform <- function(value) {
      r <- value[4]/value[2]
      c(theta=value[1],rho=r,mediated=r*value[1],gamma=value[3]-r*value[1],total=value[3])
    }
    gradient <- rbind(c(1,0,rho,-rho,0),
      c(0,-rho/be[g_E],-theta*rho/be[g_E],theta*rho/be[g_E],0),
      c(0,0,0,1,1),c(0,1/be[g_E],theta/be[g_E],-theta/be[g_E],0))
    numeric_gradient <- matrix(0,4,5)
    for(j in seq_along(rf_coefficients)) {
      step <- 1e-6*max(abs(rf_coefficients[j]),.01)
      upper <- lower <- rf_coefficients; upper[j] <- upper[j]+step; lower[j] <- lower[j]-step
      numeric_gradient[j,] <- (transform(upper)-transform(lower))/(2*step)
    }
    gradient_error <- scaled_error(gradient,numeric_gradient)
    coefficient_influence <- cbind(he[,t_E],he[,g_E],hy[,t_Y],hy[,g_Y])
    derived_influence <- cbind(he[,t_E],h_rho,h_mediated,h_gamma,hy[,t_Y])
    delta_error <- scaled_error(crossprod(derived_influence),
      t(gradient)%*%crossprod(coefficient_influence)%*%gradient)
    influence_identity_error <- max(abs(h_gamma+h_mediated-hy[,t_Y]))
    point_identity_error <- abs(row$gamma_estimate+row$mediated_estimate-total)
    require_ok(embedding_error<1e-10 && normal_E_error<1e-9 && coefficient_E_error<1e-8 &&
      covariance_error<1e-8 && cross_error<1e-8 && symmetry_error<1e-12 &&
      gradient_error<1e-6 && delta_error<1e-10 && influence_identity_error<1e-10 &&
      point_identity_error<1e-10,'Model numerical validation failed.')
    diagnostics[[measure]] <- c(list(
      normal_equations_RF_E_scaled_error=normal_E_error,
      normal_equations_RF_Y_scaled_error=normal_Y_error,
      independent_RF_E_coefficient_scaled_error=coefficient_E_error,
      independent_RF_Y_coefficient_scaled_error=coefficient_Y_error,
      restricted_full_design_prediction_error=embedding_error,
      omitted_proxy_coefficients_fixed_zero=all(embedded[W]==0),
      independent_CR1_covariance_scaled_error=covariance_error,
      independent_cross_equation_covariance_scaled_error=cross_error,
      joint_covariance_symmetry_error=symmetry_error,
      joint_covariance_minimum_eigenvalue=minimum_eigenvalue,
      finite_difference_gradient_scaled_error=gradient_error,
      delta_covariance_scaled_error=delta_error,
      gamma_plus_mediated_influence_error=influence_identity_error,
      gamma_plus_mediated_point_error=point_identity_error),fieller$checks)

    for(equation in c('RF_E','RF_Y')) {
      fit <- if(equation=='RF_E') fe else fy
      design <- if(equation=='RF_E') x_E else x_Y
      correction <- if(equation=='RF_E') correction_E else correction_Y
      for(j in seq_len(ncol(design))) reduced[[length(reduced)+1L]] <- as.data.frame(c(
        list(measure_id=measure,equation=equation,term=colnames(design)[j]),
        stats_row(fit$b[j],sqrt(fit$covariance[j,j]),df),
        list(n=n,classrooms=groups,df=df,design_rank=ncol(design),CR1_multiplier=correction)),
        check.names=FALSE)
    }
    rows[[m]] <- as.data.frame(row,check.names=FALSE)
  }
  comparison <- do.call(rbind,rows); reduced <- do.call(rbind,reduced)
  alternatives <- comparison$measure_id!='E_baseline'
  for(inference in c('CR1','CR2')) {
    column <- if(inference=='CR1') 'theta_p_t' else 'theta_p_CR2'
    name <- paste0('theta_p_holm3_',inference)
    comparison[[name]] <- NA_real_
    comparison[[name]][alternatives] <- p.adjust(comparison[[column]][alternatives],method='holm',n=3)
  }
  table <- paper_table(comparison)
  list(comparison=comparison,reduced_forms=reduced,paper_table=table,
    table_markdown=table_markdown(table),table_latex=table_latex(table),metadata=list(
      specification='equation_specific_reduced_forms',model_requests=0,
      empirical_model=c('E=pi0+theta*T+piG*G+piD*D+xi',
        "Y=delta0+rho*E+gamma*T+deltaD*D+deltaW'W+eta"),
      estimator='Separate reduced forms with a restricted four-column effort-proxy first stage',
      observations=n,students=468,classrooms=groups,measure_order=MEASURES,
      regressor_order_RF_E=colnames(x_E),regressor_order_RF_Y=colnames(x_Y),
      design_rank_RF_E=k_E,design_rank_RF_Y=k_Y,
      structural_outcome_W=W,structural_outcome_difficulty='D',structural_outcome_excluded='G',
      model_implied_proxy_zero_restrictions=list(terms=W,fixed_values=rep(0,length(W))),
      instrument='Prior GPA G',
      sample_policy='1163 student-sessions; equal weights',
      CR1_correction='C/(C-1) * (N-1)/(N-K_j), computed separately for each equation j',
      CR1_multiplier_RF_E=correction_E,CR1_multiplier_RF_Y=correction_Y,
      cross_equation_multiplier=sqrt(correction_E*correction_Y),CR1_t_df=df,
      CR2_theta=c(cr2$metadata,list(design='RF_E: intercept, T, D, G',design_rank=k_E)),
      decomposition='theta=piET; rho=piYG/piEG; mediated=rho*piET; gamma=piYT-mediated',
      ratio_confidence_set='Fieller; joint GPA-coefficient pair; t(26)',
      maintained_assumptions=c('Linear equations and the same positive effort-to-proxy relation across arms',
        'GPA exclusion and orthogonality of all observed regressors to each structural error',
        'Common effective-effort measurement relation across treatment arms'),
      Holm_family='Three alternative-measure theta tests; CR1 and CR2 adjusted separately',
      numerical_validation=diagnostics,
      checks=list(complete_common_sample=TRUE,designs_full_rank=TRUE,
        proxy_four_column_restriction_applied=TRUE,restricted_embedding_checks_passed=TRUE,
        separate_CR1_corrections_used=TRUE,cross_equation_covariance_checked=TRUE,
        joint_covariance_PSD_checked=TRUE,finite_difference_delta_checks_passed=TRUE,
        Fieller_polynomial_and_endpoint_checks_passed=TRUE,
        CR2_four_column_design_checks_passed=TRUE)))
}

# JSON serialization.
json_quote <- function(value) {
  special <- c('34'=r"(\")",'92'=r"(\\)",'8'=r"(\b)",'9'=r"(\t)",
               '10'=r"(\n)",'12'=r"(\f)",'13'=r"(\r)")
  escaped <- vapply(utf8ToInt(enc2utf8(value)),function(code) {
    name <- as.character(code)
    if(name %in% names(special)) special[[name]] else if(code<32) sprintf('\\u%04x',code) else intToUtf8(code)
  },character(1))
  paste0('"',paste0(escaped,collapse=''),'"')
}
to_json <- function(value) {
  if(is.null(value)) return('null')
  if(is.list(value)) {
    if(!is.null(names(value)) && all(nzchar(names(value))))
      return(paste0('{',paste(vapply(seq_along(value),function(i)
        paste0(json_quote(names(value)[i]),':',to_json(value[[i]])),character(1)),collapse=','),'}'))
    return(paste0('[',paste(vapply(value,to_json,character(1)),collapse=','),']'))
  }
  if(length(value)!=1L) return(paste0('[',paste(vapply(as.list(value),to_json,character(1)),collapse=','),']'))
  if(is.na(value)) return('null')
  if(is.logical(value)) return(if(value) 'true' else 'false')
  if(is.numeric(value)) return(if(is.finite(value)) sprintf('%.17g',value) else 'null')
  json_quote(as.character(value))
}

parse_arguments <- function(args,package) {
  result <- list(input=file.path(package,'data','clean','analysis_sessions.csv'),output_dir=file.path(package,'results'))
  i <- 1L
  while(i<=length(args)) {
    flag <- args[i]
    if(flag %in% c('-h','--help')) {
      cat('Usage: Rscript code/run_regressions.R [--input PATH] [--output-dir PATH]\n')
      cat('Reads only the complete analysis CSV; writes regression estimates, tables, and regression_info.json.\n')
      return(NULL)
    }
    require_ok(flag %in% c('--input','--output-dir') && i<length(args),paste('Unknown or incomplete argument:',flag))
    result[[if(flag=='--input') 'input' else 'output_dir']] <- args[i+1L]
    i <- i+2L
  }
  result
}


main <- function() {
  start <- proc.time()[['elapsed']]
  started <- format(Sys.time(),'%Y-%m-%dT%H:%M:%OS3Z',tz='UTC')
  file_argument <- grep('^--file=',commandArgs(FALSE),value=TRUE)
  require_ok(length(file_argument)==1L,'Run this file using Rscript.')
  script <- normalizePath(sub('^--file=','',file_argument),winslash='/',mustWork=TRUE)
  package <- dirname(dirname(script))
  args <- parse_arguments(commandArgs(TRUE),package)
  if(is.null(args)) return(invisible(NULL))
  input <- normalizePath(args$input,winslash='/',mustWork=TRUE)
  input_hash <- unname(tools::md5sum(input))
  analysis <- read.csv(input,check.names=FALSE,encoding='UTF-8')
  primary <- estimate_models(analysis)
  output <- args$output_dir
  dir.create(output,recursive=TRUE,showWarnings=FALSE)
  output <- normalizePath(output,winslash='/',mustWork=TRUE)
  write_output <- function(frame,name) write.csv(frame,file.path(output,name),row.names=FALSE,na='',fileEncoding='UTF-8')
  write_output(primary$comparison,'main_estimates.csv')
  write_output(primary$paper_table,'regression_diagnostics.csv')
  writeLines(primary$table_latex,file.path(output,'regression_diagnostics.tex'),useBytes=TRUE)
  writeLines(primary$table_markdown,file.path(output,'regression_diagnostics.md'),useBytes=TRUE)
  write_output(primary$reduced_forms,'reduced_form_coefficients.csv')
  require_ok(identical(input_hash,unname(tools::md5sum(input))),'The analysis input changed during regression.')
  info <- list(status='passed',started_utc=started,
    completed_utc=format(Sys.time(),'%Y-%m-%dT%H:%M:%OS3Z',tz='UTC'),
    elapsed_seconds=proc.time()[['elapsed']]-start,input_csv=input,input_md5=input_hash,
    input_unchanged=TRUE,output_dir=output,model_requests=0,
    counts=list(constructions=nrow(primary$comparison),observations=nrow(analysis),
      students=length(unique(analysis$student_id)),classrooms=length(unique(analysis$class_id)),
      reduced_form_rows=nrow(primary$reduced_forms)),
    settings=primary$metadata,checks=primary$metadata$checks,
    environment=list(R=R.version.string,platform=R.version$platform,
      external_R_packages=character(0),regression_language='R'))
  writeLines(to_json(info),file.path(output,'regression_info.json'),useBytes=TRUE)
  cat('Completed: four dialogue measurements, GPA instrument,',nrow(analysis),'observations; all checks passed.\n')
  print(primary$comparison[c('measure_id','theta_estimate','theta_p_CR2',
    'rho_estimate','rho_p_t','gamma_estimate','gamma_p_t')],row.names=FALSE)
  cat('Results:',output,'\n')
  invisible(info)
}

if(sys.nframe()==0L) main()
