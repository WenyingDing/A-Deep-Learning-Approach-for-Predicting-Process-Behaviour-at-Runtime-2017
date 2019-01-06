<xsl:stylesheet version="2.0" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xes="http://www.xes-standard.org/">

<xsl:output method="text"/>

<xsl:template match="*|/">
  <xsl:apply-templates/>
</xsl:template>

<xsl:template match="text()|@*">
  <!--<xsl:value-of select="."/> -->
</xsl:template>


<xsl:template match="xes:log">


  <xsl:message terminate="no">
<!-- this is for XSLT 1.0 (very slow)
	<xsl:value-of select="count(//xes:string[@key='org:resource']/@value[not(following::xes:string[@key='org:resource']/@value = .)])"/>
-->
	<xsl:text>Number of traces: </xsl:text><xsl:value-of select="count(//xes:trace)"/><xsl:text>
</xsl:text>
	<xsl:text>Distinct org resources: </xsl:text><xsl:value-of select="count(distinct-values(//xes:event/xes:string[@key='org:resource']/@value))"/><xsl:text>
</xsl:text>
	<xsl:text>Distinct org groups: </xsl:text><xsl:value-of select="count(distinct-values(//xes:event/xes:string[@key='org:group']/@value))"/><xsl:text>
</xsl:text>
	<xsl:text>Distinct event names: </xsl:text><xsl:value-of select="count(distinct-values(//xes:event/xes:string[@key='concept:name']/@value))"/><xsl:text>
</xsl:text>
	<xsl:text>Distinct lifecycle transitions: </xsl:text><xsl:value-of select="count(distinct-values(//xes:event/xes:string[@key='lifecycle:transition']/@value))"/><xsl:text>: </xsl:text>
	<xsl:for-each select="distinct-values(//xes:event/xes:string[@key='lifecycle:transition']/@value)">
		<xsl:value-of select="."/><xsl:text>, </xsl:text>
	</xsl:for-each>

  </xsl:message>

</xsl:template>

</xsl:stylesheet>

