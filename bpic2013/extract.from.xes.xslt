<xsl:stylesheet version="1.0" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xes="http://www.xes-standard.org/">

<xsl:output method="text"/>

<xsl:template match="*|/">
  <xsl:apply-templates/>
</xsl:template>

<xsl:template match="text()|@*">
  <!--<xsl:value-of select="."/> -->
</xsl:template>


<xsl:template match="xes:trace">
  <xsl:message terminate="no">

    <xsl:text>Found trace </xsl:text>
    <xsl:value-of select="xes:string[@key='concept:name']/@value"/>
    <xsl:text> with </xsl:text>
    <xsl:value-of select="count(xes:event)"/>
    <xsl:text> events</xsl:text>
  </xsl:message>

  <xsl:if test="count(xes:event) &gt; 0">
    <xsl:apply-templates select="xes:event">
      <xsl:sort select="xes:string[@key='time:timestamp']/@value"/>
    </xsl:apply-templates>

    <xsl:text> [EOC]
</xsl:text>
  </xsl:if>
</xsl:template>

<xsl:template match="xes:event">
  <xsl:value-of select="translate(xes:string[@key='concept:name']/@value, '_-. ', '')"/><xsl:text>-</xsl:text><xsl:value-of select="translate(xes:string[@key='lifecycle:transition']/@value, '_-. ', '')"/>
  <xsl:text> </xsl:text>
</xsl:template>

</xsl:stylesheet>

